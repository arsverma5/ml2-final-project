import streamlit as st
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from src.model import fit_sarima, forecast_sarima


try:
    model = joblib.load("results/model.joblib")
except:
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
    st.header("Neural Network")

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
        pred_log = model.predict(input_df)[0]
        # reverse log transform
        pred = np.expm1(pred_log)
        st.success(f"Predicted Resolution Time: {pred:.2f} hours")

with tab3:
    st.header("Bayesian")

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
        weekly = pd.read_csv('data/weekly_counts.csv', parse_dates=['created_date'])
        weekly_res = pd.read_csv('data/weekly_resolution.csv', parse_dates=['created_date'])
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

   