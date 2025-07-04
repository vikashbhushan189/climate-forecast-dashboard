import numpy as np 
import pandas as pd 
import datetime as dt
import pickle
import os

from sklearn import linear_model
from statsmodels.tsa.statespace.sarimax import SARIMAX
from prophet import Prophet

if not os.path.exists('models'):
    os.makedirs('models')

# --- PART 1: CO₂ MODELS (No changes here) ---
print("--- Starting CO₂ Model Training ---")
CO2_df = pd.read_csv('data/data/archive.csv')
CO2_df['Date'] = pd.to_datetime(CO2_df['Year'].astype(str) + '-' + CO2_df['Month'].astype(str) + '-01')
CO2_df.dropna(subset=['Carbon Dioxide (ppm)'], inplace=True)
co2_data = CO2_df.set_index('Date')['Carbon Dioxide (ppm)'].resample('ME').mean()
co2_data.interpolate(method='time', inplace=True)
co2_data.to_csv('models/cleaned_co2_data.csv')
print("Cleaned CO₂ data saved.")

lr_model_co2 = linear_model.LinearRegression().fit(
    [[d.year, d.month, d.month**2, d.year**2] for d in co2_data.index], co2_data.values)
pickle.dump(lr_model_co2, open('models/co2_linear_regression.pkl', 'wb'))
print("CO₂ Linear Regression model saved.")

sarima_model_co2 = SARIMAX(co2_data, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12)).fit(disp=False)
pickle.dump(sarima_model_co2, open('models/co2_sarima_model.pkl', 'wb'))
print("CO₂ SARIMA model saved.")

prophet_df_co2 = co2_data.reset_index().rename(columns={'Date': 'ds', 'Carbon Dioxide (ppm)': 'y'})
prophet_model_co2 = Prophet(growth='linear', yearly_seasonality=True).fit(prophet_df_co2)
pickle.dump(prophet_model_co2, open('models/co2_prophet_model.pkl', 'wb'))
print("CO₂ Prophet model saved.")

print("\n--- CO₂ Model Training Complete ---\n")


# --- PART 2: TEMPERATURE MODELS (Changes are here) ---
print("--- Starting Global Temperature Model Training ---")
Temp_df = pd.read_csv('berkeleyearth/GlobalTemperatures.csv')
Temp_df['dt'] = pd.to_datetime(Temp_df['dt'])
Temp_df.dropna(subset=['LandAverageTemperature'], inplace=True)
temp_data = Temp_df.set_index('dt')['LandAverageTemperature'].resample('ME').mean()
temp_data.interpolate(method='time', inplace=True)

# ==============================================================================
# --- THE CRITICAL FIX ---
# Truncate the data to start from 1960. This solves the OverflowError and
# makes the model more relevant to modern climate trends.
temp_data = temp_data.loc['1960-01-01':]
print("Temperature data truncated to start from 1960 for relevance and stability.")
# ==============================================================================

temp_data.to_csv('models/cleaned_temp_data.csv')
print("Cleaned Temperature data saved.")

lr_model_temp = linear_model.LinearRegression().fit(
    [[d.year, d.month, d.month**2, d.year**2] for d in temp_data.index], temp_data.values)
pickle.dump(lr_model_temp, open('models/temperature_linear_regression.pkl', 'wb'))
print("Temperature Linear Regression model saved.")

sarima_model_temp = SARIMAX(temp_data, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12)).fit(disp=False)
pickle.dump(sarima_model_temp, open('models/temperature_sarima_model.pkl', 'wb'))
print("Temperature SARIMA model saved.")

prophet_df_temp = temp_data.reset_index().rename(columns={'dt': 'ds', 'LandAverageTemperature': 'y'})
prophet_model_temp = Prophet(growth='linear', yearly_seasonality=True).fit(prophet_df_temp)
pickle.dump(prophet_model_temp, open('models/temperature_prophet_model.pkl', 'wb'))
print("Temperature Prophet model saved.")

print("\n--- Global Temperature Model Training Complete ---\n")
print("--- All models have been trained and saved successfully! ---")