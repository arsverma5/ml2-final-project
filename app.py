import streamlit as st
import joblib
import pandas as pd
import numpy as np

model = joblib.load("results/model.joblib")

tab1, tab2, tab3, tab4 = st.tabs(["Home", " Neural Network", "Bayesian", "Time Series"])

with tab1:
    st.header("Home")
    st.write("Write project description here.")

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

    hour = st.slider("Hour of Day (0–23)", 0, 23, 10)
    day_of_week = st.slider("Day of Week (1=Mon, 7=Sun)", 1, 7, 2)
    month = st.slider("Month (1=Jan, 12=Dec)", 1, 12, 1)

    st.write("Inputs Preview:")
    st.write({
        "Complaint.Type": complaint_type,
        "Agency": agency,
        "Police.Precinct": police_precinct,
        "Borough": borough,
        "Location.Type": location_type,
        "incident_zip": incident_zip,
        "Hour": hour,
        "Day.Of.Week": day_of_week,
        "Month": month,
    })

    input_df = pd.DataFrame([{
        "agency": agency,
        "complaint_type": complaint_type,
        "location_type": location_type,
        "police_precinct": police_precinct,
        "borough": borough,
        "hour": hour,
        "day_of_week": day_of_week,
        "month": month,
        "incident_zip": incident_zip
    }]) 

    input_df['hour_sin'] = np.sin(2 * np.pi * input_df['hour'] / 24)
    input_df['hour_cos'] = np.cos(2 * np.pi * input_df['hour'] / 24)

    input_df['dow_sin'] = np.sin(2 * np.pi * input_df['day_of_week'] / 7)
    input_df['dow_cos'] = np.cos(2 * np.pi * input_df['day_of_week'] / 7)

    input_df['month_sin'] = np.sin(2 * np.pi * input_df['month'] / 12)
    input_df['month_cos'] = np.cos(2 * np.pi * input_df['month'] / 12)

    input_df = input_df.drop(columns=["hour", "day_of_week", "month"])

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
    st.header("Time Series")