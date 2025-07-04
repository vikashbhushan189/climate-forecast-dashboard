import streamlit as st
import pandas as pd
import pickle
import plotly.graph_objs as go
import datetime as dt

# --- App Configuration ---
st.set_page_config(
    page_title="Climate Forecast Dashboard",
    layout="wide"
)

# --- Caching Function (New, Robust Structure) ---
@st.cache_data
def get_forecast_data(target_key):
    """
    This is the master function that loads data and models, and generates
    the forecast DataFrame. Caching is based on the target_key.
    """
    st.write(f"Cache miss: Running full forecast for '{target_key}'...")

    # --- 1. Load Data and Models based on target_key ---
    if target_key == "co2":
        data = pd.read_csv('models/cleaned_co2_data.csv', index_col='Date', parse_dates=True)
        data = data.rename(columns={'Carbon Dioxide (ppm)': 'value'})
        lr_model = pickle.load(open('models/co2_linear_regression.pkl', 'rb'))
        sarima_model = pickle.load(open('models/co2_sarima_model.pkl', 'rb'))
        prophet_model = pickle.load(open('models/co2_prophet_model.pkl', 'rb'))
    else: # target_key == "temp"
        data = pd.read_csv('models/cleaned_temp_data.csv', index_col='dt', parse_dates=True)
        data = data.rename(columns={'LandAverageTemperature': 'value'})
        lr_model = pickle.load(open('models/temperature_linear_regression.pkl', 'rb'))
        sarima_model = pickle.load(open('models/temperature_sarima_model.pkl', 'rb'))
        prophet_model = pickle.load(open('models/temperature_prophet_model.pkl', 'rb'))
    
    data = data.asfreq('ME')

    # --- 2. Create Forecast DataFrame ---
    historical_df = pd.DataFrame({'Historical': data['value']})
    
    forecast_start_date = data.index.max() + pd.DateOffset(months=1)
    forecast_end_date = dt.datetime(2050, 12, 31)
    forecast_dates = pd.date_range(start=forecast_start_date, end=forecast_end_date, freq='ME')

    future_df = pd.DataFrame(index=forecast_dates)

    lr_x = [[d.year, d.month, d.month**2, d.year**2] for d in forecast_dates]
    future_df['Linear Regression'] = lr_model.predict(lr_x)

    sarima_pred = sarima_model.get_forecast(steps=len(forecast_dates)).predicted_mean
    future_df['SARIMA'] = sarima_pred.values

    future_df_prophet = pd.DataFrame({'ds': forecast_dates})
    prophet_pred = prophet_model.predict(future_df_prophet).set_index('ds')['yhat']
    future_df['Prophet'] = prophet_pred.values
    
    # --- 3. Return the final combined DataFrame ---
    return pd.concat([historical_df, future_df])

# --- UI Elements ---
st.sidebar.header("Forecasting Options")
forecast_target_name = st.sidebar.radio(
    "Select Forecast Target",
    ("CO₂ Concentration", "Global Temperature")
)

# --- Main App Logic ---
if forecast_target_name == "CO₂ Concentration":
    st.title("Atmospheric CO₂ Concentration Forecast 📈")
    target_key = "co2"
    unit = "ppm"
else:
    st.title("Global Land Temperature Forecast 🌡️")
    target_key = "temp"
    unit = "°C"

# Get the correct, cached forecast DataFrame based on the selected target
forecast_df = get_forecast_data(target_key)
# Define a separate variable for historical data for the date picker min_value
historical_data = forecast_df['Historical'].dropna()

# Date input from the user
default_future_date = dt.date(2030, 1, 1)
future_date = st.sidebar.date_input(
    "Select a future date to forecast:",
    default_future_date,
    min_value=historical_data.index.max().date() + dt.timedelta(days=31),
    max_value=dt.date(2050, 12, 31)
)

# --- Display Predictions ---
st.subheader(f"Predictions for {future_date.strftime('%B %Y')}")
lookup_date = pd.to_datetime(future_date) + pd.offsets.MonthEnd(0)

try:
    predictions = forecast_df.loc[lookup_date]
    col1, col2, col3 = st.columns(3)
    col1.metric("Linear Regression", f"{predictions.get('Linear Regression', 0):.2f} {unit}")
    col2.metric("SARIMA", f"{predictions.get('SARIMA', 0):.2f} {unit}")
    col3.metric("Prophet", f"{predictions.get('Prophet', 0):.2f} {unit}")
except KeyError:
    st.error("Could not find a forecast for the selected date. Please try another.")

# --- Display Forecast Plot ---
st.subheader("Historical Data and Model Forecasts")
y_axis_title = f"Value ({unit})"
plot_title = f"{forecast_target_name}: Historical Data vs. Model Forecasts to 2050"

fig = go.Figure()
model_colors = {'Historical': 'black', 'Linear Regression': 'blue', 'SARIMA': 'red', 'Prophet': 'green'}
for column, color in model_colors.items():
    fig.add_trace(go.Scatter(
        x=forecast_df.index, y=forecast_df[column], mode='lines', name=column, 
        line=dict(color=color, dash=('dash' if column != 'Historical' else 'solid'))
    ))
fig.update_layout(
    title=plot_title,
    xaxis_title="Year", yaxis_title=y_axis_title, legend=dict(x=0.01, y=0.99)
)
st.plotly_chart(fig, use_container_width=True)