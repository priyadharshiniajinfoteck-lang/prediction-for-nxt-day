import streamlit as st
import pandas as pd
import numpy as np
from prophet import Prophet
import holidays
import plotly.graph_objects as go
from datetime import timedelta

# --- 1. App Configuration ---
st.set_page_config(page_title="Supermarket Sales Predictor", layout="wide")
st.title("🍎 Vegetable & Fruit Sales Forecasting")
st.markdown("Predicting next-day sales using **Facebook Prophet** with External Data Integration.")

# --- 2. Data Loading ---
@st.cache_data
def load_data(file):
    df = pd.read_csv(file)
    df['ds'] = pd.to_datetime(df['Transaction Date'])
    # Aggregate sales by date
    df_daily = df.groupby('ds').agg({'Sales': 'sum'}).reset_index()
    df_daily.columns = ['ds', 'y']
    return df_daily

uploaded_file = st.sidebar.file_uploader("Upload Supermarket CSV", type=["csv"])

if uploaded_file:
    data = load_data(uploaded_file)
    
    # --- 3. Feature Engineering (External Data) ---
    # Adding Holidays/Festivals (Open Source)
    country_holidays = holidays.India(years=[2024, 2025]) # Adjust country as needed
    data['is_holiday'] = data['ds'].apply(lambda x: 1 if x in country_holidays else 0)
    
    # Simulating Weather Data (Open Source Weather API integration point)
    # In a real scenario, use 'requests' to fetch from OpenWeatherMap
    np.random.seed(42)
    data['temp_avg'] = np.random.normal(25, 5, len(data)) # Dummy weather regressor
    
    # --- 4. Model Training (Prophet) ---
    model = Prophet(yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=False)
    
    # Add External Regressors for High Accuracy
    model.add_regressor('temp_avg')
    model.add_country_holidays(country_name='IN') # Built-in Prophet holidays
    
    model.fit(data)
    
    # --- 5. Predict for Next Day ---
    last_date = data['ds'].max()
    next_day = last_date + timedelta(days=1)
    
    future = model.make_future_dataframe(periods=7) # Predict 1 week ahead
    future['temp_avg'] = np.random.normal(25, 5, len(future)) # Add forecast weather
    
    forecast = model.predict(future)
    
    # --- 6. Results & Visualization ---
    col1, col2 = st.columns(2)
    
    with col1:
        next_day_val = forecast[forecast['ds'] == next_day]['yhat'].values[0]
        st.metric(label=f"Predicted Sales for {next_day.date()}", 
                  value=f"₹{next_day_val:,.2f}")
    
    with col2:
        st.write("### Sales Trend & Forecast")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=data['ds'], y=data['y'], name='Actual Sales'))
        fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat'], name='Predicted Sales', line=dict(dash='dash')))
        st.plotly_chart(fig, use_container_view=True)

    st.write("### Forecast Components (Seasonality & Holidays)")
    fig_components = model.plot_components(forecast)
    st.pyplot(fig_components)
    
else:
    st.info("Please upload the supermarket CSV file to start forecasting.")