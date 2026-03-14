# NYC 311 Request Delay Prediction — Phase I Proof of Concept

## Problem
Predict whether a NYC 311 service request will experience an unusually long resolution time based on request characteristics.

## Dataset
NYC Open Data: 311 Service Requests

## Phase I Proof of Concept

This repository contains:

we focused on data, cleaning, exploratory analysis, and establishing a baseline modeling pipeline.

Specifically, we have:

- Collected a sample of NYC 311 Service Request data from NYC Open Data
- Computed resolution time for each request using Created Date and Closed Date
- Removed requests without valid closure information
- Filtered out invalid or extreme resolution times (negative values and top 1% outliers)
- Dropped columns with excessive missing values or low relevance to the initial model
- Explored request volume across complaint types
- Constructed a cleaned dataset of closed requests suitable for modeling
- Outlined a baseline classification approach to distinguish long vs. typical resolution times

## How to Run


1. Download the data:
   - Go to https://www.kaggle.com/datasets/new-york-city/ny-311-service-requests
   - Click **Download** - this will download a `.zip` file
   - Unzip the file and place `311-service-requests-from-2010-to-present.csv` in the `data/` folder

2. Install dependencies:

pip install -r requirements.txt

3. Open notebook:

notebooks/phase1_poc.ipynb

4. Run all cells

## Repository Structure

- data/ — sample dataset
- notebooks/ — exploratory work + POC model
- src/ — preprocessing and modeling scripts, for later phases
- results/ — saved outputs

