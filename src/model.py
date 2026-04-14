import numpy as np 

# fit sarima function
def fit_sarima(y, p=1, d=1, q=1, P=1, D=1, Q=1, s=52):
    y = y.astype(float)
    
    y_diff = np.diff(y, n=d)
    
    y_sdiff = y_diff[s:] - y_diff[:-s]
    
    n = len(y_sdiff)
    max_lag = max(p, q, s*P, s*Q)
    start = max_lag
    
    if start >= n:
        raise ValueError("Not enough data for these parameters")
    
    X_rows = []
    y_target = []
    
    for t in range(start, n):
        row = []
        # AR terms
        for i in range(1, p+1):
            row.append(y_sdiff[t-i])
        # SAR terms
        for i in range(1, P+1):
            row.append(y_sdiff[t - i*s] if t - i*s >= 0 else 0)
        X_rows.append(row)
        y_target.append(y_sdiff[t])
    
    X = np.array(X_rows)
    y_target = np.array(y_target)
    
    XtX = X.T @ X + np.eye(X.shape[1]) * 1e-8
    coefs = np.linalg.solve(XtX, X.T @ y_target)
    
    residuals = y_target - X @ coefs
    
    X_ma_rows = []
    ma_target = []
    for t in range(max(q, Q*s), len(residuals)):
        row = []
        for i in range(1, q+1):
            row.append(residuals[t-i])
        for i in range(1, Q+1):
            idx = t - i*s
            row.append(residuals[idx] if idx >= 0 else 0)
        X_ma_rows.append(row)
        ma_target.append(residuals[t])
    
    X_ma = np.array(X_ma_rows)
    ma_target = np.array(ma_target)
    XtX_ma = X_ma.T @ X_ma + np.eye(X_ma.shape[1]) * 1e-8
    ma_coefs = np.linalg.solve(XtX_ma, X_ma.T @ ma_target)
    
    return coefs, ma_coefs, y_sdiff, y_diff, residuals


# forecast sarima function
def forecast_sarima(y, coefs, ma_coefs, residuals, p=1, d=1, q=1, P=1, D=1, Q=1, s=52, n_steps=52):
    y = y.astype(float)
    y_diff = list(np.diff(y, n=d))
    y_sdiff = list(np.array(y_diff[s:]) - np.array(y_diff[:-s]))
    res = list(residuals)
    
    future_sdiff = []
    
    for _ in range(n_steps):
        n = len(y_sdiff)
        row = []
        for i in range(1, p+1):
            row.append(y_sdiff[n-i])
        for i in range(1, P+1):
            idx = n - i*s
            row.append(y_sdiff[idx] if idx >= 0 else 0)
        
        ar_pred = np.dot(row, coefs)
        
        ma_row = []
        for i in range(1, q+1):
            ma_row.append(res[-i])
        for i in range(1, Q+1):
            idx = len(res) - i*s
            ma_row.append(res[idx] if idx >= 0 else 0)
        
        ma_pred = np.dot(ma_row, ma_coefs)
        
        next_sdiff = ar_pred + ma_pred
        future_sdiff.append(next_sdiff)
        y_sdiff.append(next_sdiff)
        res.append(0.0)
    
 
    full_sdiff = list(np.array(y_diff[s:]) - np.array(y_diff[:-s])) + future_sdiff
    
    full_ydiff = list(y_diff[:s])
    for i in range(len(full_sdiff)):
        full_ydiff.append(full_sdiff[i] + full_ydiff[-s])
    
    full_y = [y[0]]
    for diff_val in full_ydiff:
        full_y.append(full_y[-1] + diff_val)
    
    future_vals = full_y[-n_steps:]
    return np.array(future_vals)

