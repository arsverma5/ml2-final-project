import streamlit as st
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from src.model import fit_sarima, forecast_sarima

MODEL_PATH = Path(__file__).resolve().parent / "nn_model.joblib"

@st.cache_resource
def load_model(path: Path):
    return joblib.load(path)

try:
    model = load_model(MODEL_PATH)
except Exception as exc:
    st.error(f"Failed to load model: {exc}")
    model = None

tab1, tab2, tab3, tab4 = st.tabs(["Home", " Neural Network", "Bayesian", "Time Series"])

with tab1:
    st.title("Predicting NYC 311 Service Request Resolution Times")
    st.subheader("A Bayesian, Neural Network, and Time Series Approach")
    st.markdown("---")

    st.write("""
    New York City's 311 system receives millions of service requests every year, 
    spanning everything from noise complaints and heating failures to rodent sightings 
    and illegal parking. While the system provides residents with a unified channel for 
    reporting non-emergency issues, response times vary widely: some complaints are 
    resolved within hours, others take weeks.
    
    This project applies three machine learning methods to predict resolution times 
    and identify when the city is most likely to fall behind on service requests.
    """)

    st.markdown("---")
    st.subheader("Methods")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.info("**Bayesian Regression**\n\nPredicts individual resolution time from complaint type, borough, and time features.")
    with col2:
        st.info("**Neural Network**\n\nMLP with 3 hidden layers predicting log resolution hours across 10 complaint types.")
    with col3:
        st.info("**Time Series** \n\nForecasts weekly complaint volume and resolution time by complaint type.")
 
    
with tab2:
    st.header("Neural Network: Predict Resolution Time")

    st.write("""
    This page uses a trained neural network to predict NYC 311
    service request resolution time from complaint and location features.
    Predictions are made in log-hours and then converted back to actual hours.
    """)

    st.subheader("Model Inputs")

    complaint_type = st.selectbox(
        "Complaint Type",
        ['HEAT/HOT WATER', 'Illegal Parking', 'Noise - Street/Sidewalk',
         'Blocked Driveway', 'Noise - Residential', 'Noise - Commercial',
         'PLUMBING', 'Request Large Bulky Item Collection',
         'Noise - Vehicle', 'UNSANITARY CONDITION']
    )

    agency = st.selectbox(
        "Agency",
        ['HPD', 'NYPD', 'DSNY']
    )

    police_precinct = st.number_input("Police Precinct", min_value=0, value=1)

    borough = st.selectbox(
        "Borough",
        ['BROOKLYN', 'QUEENS', 'MANHATTAN', 'BRONX', 'STATEN ISLAND', 'Unspecified']
    )


    location_type = st.selectbox(
        "Location Type",
        ['RESIDENTIAL BUILDING', 'Street/Sidewalk',
         'Residential Building/House', 'Store/Commercial',
         'Club/Bar/Restaurant', 'Sidewalk']
    )

    incident_zip = st.number_input("Incident Zip", min_value=0, value=11201)

    st.write("Inputs Preview:")
    st.write({
        "Complaint.Type": complaint_type,
        "Agency": agency,
        "Police.Precinct": police_precinct,
        "Borough": borough,
        "Location.Type": location_type,
        "Incident.Zip": incident_zip,
    })

    input_df = pd.DataFrame([{
        "agency": agency,
        "complaint_type": complaint_type,
        "location_type": location_type,
        "police_precinct": police_precinct,
        "borough": borough,
        "incident_zip": incident_zip
    }]) 

    cat_cols = ["agency", "complaint_type", "location_type", "borough", "police_precinct"]

    # ensure in string format
    for col in cat_cols:
        input_df[col] = input_df[col].astype(str)

    if st.button("Predict Resolution Time (NN)"):
        if model is None:
            st.error("The neural network model is not available.")
        else:
            pred_log = model.predict(input_df)[0]
            # reverse log transform
            pred = np.expm1(pred_log)
            st.success(f"Predicted Resolution Time: {pred:.2f} hours")

with tab3:
    st.header("Bayesian Model: Predict Resolution Time")

    st.write("""
    This tool uses the actual posterior draws from our Bayesian regression model 
    (fit in R using brms) to generate a posterior predictive distribution for resolution time.
    Instead of a single prediction, you get a full distribution of plausible outcomes.
    """)

    @st.cache_data
    def load_posterior():
        fixed = pd.read_csv("results/posterior_fixed.csv")
        complaints = pd.read_csv("results/posterior_complaint.csv")
        return fixed, complaints

    post_fixed, post_complaints = load_posterior()

    st.subheader("Select Inputs")
    col1, col2 = st.columns(2)

    with col1:
        borough = st.selectbox("Borough", 
            ['BRONX', 'BROOKLYN', 'MANHATTAN', 'QUEENS', 'STATEN ISLAND'],
            key="bayes_borough")
        
        complaint_options_raw = sorted(post_complaints['Complaint.Type'].unique())
        complaint_options_display = [c.replace(".", " ").strip() for c in complaint_options_raw]
        display_to_raw = dict(zip(complaint_options_display, complaint_options_raw))

        complaint_display = st.selectbox("Complaint Type", complaint_options_display, key="bayes_complaint")
        complaint = display_to_raw[complaint_display]

    with col2:
        hour = st.slider("Hour of Day", 0, 23, 10, key="bayes_hour")
        day = st.selectbox("Day of Week",
            ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday'],
            key="bayes_day")

    if st.button("Generate Bayesian Prediction"):
        st.caption("Each bar represents how many of the 5,000 posterior draws predicted that resolution time. " \
        "The distribution captures genuine uncertainty — not just a single number.")

        borough_col = {
            'BRONX': None,
            'BROOKLYN': 'b_BoroughBROOKLYN',
            'MANHATTAN': 'b_BoroughMANHATTAN',
            'QUEENS': 'b_BoroughQUEENS',
            'STATEN ISLAND': 'b_BoroughSTATENISLAND'
        }

        complaint_re = post_complaints[post_complaints['Complaint.Type'] == complaint]['r_Complaint.Type'].values

        if len(complaint_re) == 0:
            complaint_re = np.zeros(len(post_fixed))

        n_draws = len(post_fixed)
        complaint_samples = np.random.choice(complaint_re, size=n_draws, replace=True)

        intercept = post_fixed['b_Intercept'].values
        sigma = post_fixed['sigma'].values

        if borough_col[borough] is not None:
            borough_effect = post_fixed[borough_col[borough]].values
        else:
            borough_effect = np.zeros(n_draws)

        log_pred = intercept + borough_effect + complaint_samples + np.random.normal(0, sigma)
        pred_hours = np.exp(log_pred)

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.hist(pred_hours, bins=50, color='steelblue', edgecolor='white')
        ax.set_title(f"Posterior Predictive: {complaint_display} in {borough} at {hour}:00")
        ax.set_xlabel("Predicted Resolution Hours")
        ax.set_ylabel("Frequency")
        ax.set_xlim(0, np.percentile(pred_hours, 97))
        st.pyplot(fig)

        st.success(f"Posterior Mean: {pred_hours.mean():.1f} hours")
        st.info(f"95% Credible Interval: [{np.percentile(pred_hours, 2.5):.1f}, {np.percentile(pred_hours, 97.5):.1f}] hours")  
with tab4:
    st.header("Time Series: SARIMA Forecast")
    st.write("""
    We implemented a Seasonal ARIMA (SARIMA) model from scratch using only NumPy to forecast 
    weekly NYC 311 complaint volume and average resolution time by complaint type. 
    The model specification SARIMA(1,1,1)(1,1,1,52) was selected based on stationarity 
    checks, manual ACF computation, and empirical parameter comparison. Models were trained 
    on 2020–2024 data and evaluated on a held-out 2025 test set.
    """)

    st.subheader("Generate a Live Forecast")
    st.write("Select a complaint type and target to generate a live 52-week SARIMA forecast.")

    @st.cache_data
    def load_ts_data():
        weekly = pd.read_csv('data_streamlit/weekly_counts.csv', parse_dates=['created_date'])
        weekly_res = pd.read_csv('data_streamlit/weekly_resolution.csv', parse_dates=['created_date'])
        return weekly, weekly_res

    weekly, weekly_res = load_ts_data()

    col1, col2, col3 = st.columns(3)
    with col1:
        ts_complaint = st.selectbox("Complaint Type", [
            'HEAT/HOT WATER',
            'NOISE - RESIDENTIAL',
            'NOISE - STREET/SIDEWALK',
            'RODENT'
        ], key='ts_complaint')
    with col2:
        ts_target = st.radio("Target", ["Complaint Volume", "Resolution Time"], key='ts_target')
    with col3:
        n_weeks = st.slider("Forecast Horizon (weeks)", 4, 52, 52, step=4, key='ts_weeks')

    if st.button("Generate Forecast"):
        with st.spinner("Fitting SARIMA model..."):
            if ts_target == "Complaint Volume":
                data = weekly[weekly['complaint_type'] == ts_complaint].copy()
                data = data.sort_values('created_date').reset_index(drop=True)
                y = data['count'].values.astype(float)
                ylabel = "Weekly Complaints"
            else:
                data = weekly_res[weekly_res['complaint_type'] == ts_complaint].copy()
                data = data.sort_values('created_date').reset_index(drop=True)
                y = data['avg_resolution_hours'].values.astype(float)
                ylabel = "Avg Resolution Hours"

       
            coefs, ma_coefs, _, _, residuals = fit_sarima(y)
            last_date = data['created_date'].max()
            future_dates = pd.date_range(start=last_date, periods=n_weeks+1, freq='W')[1:]
            future = np.clip(forecast_sarima(y, coefs, ma_coefs, residuals, n_steps=n_weeks), 0, None)

            fig, ax = plt.subplots(figsize=(12, 5))
            ax.plot(data['created_date'], y, label='Historical', color='steelblue', alpha=0.8)
            ax.plot(future_dates, future, label=f'{n_weeks}-Week Forecast', color='red', linestyle='--', linewidth=2)
            ax.axvline(last_date, color='gray', linestyle=':', label='Forecast Start')
            ax.set_title(f'{ts_complaint} - {ts_target} Forecast')
            ax.set_xlabel('Date')
            ax.set_ylabel(ylabel)
            ax.legend()
            plt.tight_layout()
            st.pyplot(fig)

   