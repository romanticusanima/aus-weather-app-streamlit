import streamlit as st
import pandas as pd
import numpy as np
import joblib

from utils.process_weather_data import preprocess_new_data

# ---------- Page layout ----------
st.set_page_config(
    page_title='Australian Rain Prediction',
    page_icon='🌧️',
    layout='wide',
)

st.title('🌧️ Will it Rain Tomorrow in Australia?')
st.write(
    'Use the controls in the sidebar to set today’s weather conditions, '
    'then click **Predict** to see the model’s forecast.'
)
st.image('images/aus_weather.png', width=600)

raw_df = pd.read_csv('data/weatherAUS.csv')
locations = raw_df['Location'].dropna().unique().tolist()
wind_dirs = raw_df['WindGustDir'].dropna().unique().tolist()

# ---------- Sidebar with 2 columns ----------
st.markdown("""
<style>
/* Set min and max width of the sidebar */
[data-testid="stSidebar"] {
    min-width: 50%;
}
</style>
""", unsafe_allow_html=True)

with st.sidebar:
    st.header('Weather inputs')

    col1, col2, col3 = st.columns(3)

    # --- Column 1: dropdowns ---
    with col1:
        location = st.selectbox('Location', options=locations, index=15)
        windGustDir = st.selectbox('Strongest Wind Gust Direction', wind_dirs, index=4)
        windDir9am = st.selectbox('Wind Direction at 9am', wind_dirs, index=11)
        windDir3pm = st.selectbox('Wind Direction at 3pm', wind_dirs, index=6)
        rainToday = st.selectbox('Rain today?', options=['No', 'Yes'], index=1)

    # --- Column 2: Sliders ---
    with col2:
        minTemp = st.slider('Minimum Temperature', -10., 50., 23.2, step=0.1, format='%0.1f')
        maxTemp = st.slider('Maximum Temperature', -10., 50., 33.6, step=0.1, format='%0.1f')
        temp9am = st.slider('Temperature at 9am', -10., 50., 25., step=0.1, format='%0.1f')
        temp3pm = st.slider('Temperature at 3pm', -10., 50., 31., step=0.1, format='%0.1f')
        rainfall = st.slider('Rainfall', 0., 400., 10.2, step=0.1, format='%0.1f')
        windGustSpeed = st.slider('Strongest Wind Gust Speed', 0, 140, 52, step=1)
        windSpeed9am = st.slider('Wind Speed at 9am', 0, 140, 13, step=1)
        windSpeed3pm = st.slider('Wind Speed at 3pm', 0, 140, 18, step=1)
        
    # --- Column 3: Sliders ---
    with col3:
        sunshine = st.slider('Sunshine', 0., 15., 8.4, step=0.1, format='%0.1f')
        cloud9am = st.slider('Cloud Cover at 9am', 0, 9, 8, step=1)
        cloud3pm = st.slider('Cloud Cover at 3pm', 0, 9, 5, step=1)
        evaporation = st.slider('Evaporation', 0., 150., 4.2, step=0.1, format='%0.1f')
        humidity9am = st.slider('Humidity at 9am', 0, 100, 89, step=1)
        humidity3pm = st.slider('Humidity at 3pm', 0, 100, 58, step=1)
        pressure9am = st.slider('Atmospheric Pressure at 9am', 970., 1050., 1004.2, step=0.1, format='%0.1f')
        pressure3pm = st.slider('Atmospheric Pressure at 3pm', 970., 1050., 1001.3, step=0.1, format='%0.1f')

# ---------- Prepare input dict for preprocessing ----------
input_data = {
    'Location': location,
    'MinTemp': minTemp,
    'MaxTemp': maxTemp,
    'Rainfall': rainfall,
    'Evaporation': evaporation,
    'Sunshine': sunshine,
    'WindGustDir': windGustDir,
    'WindGustSpeed': windGustSpeed,
    'WindDir9am': windDir9am,
    'WindDir3pm': windDir3pm,
    'WindSpeed9am': windSpeed9am,
    'WindSpeed3pm': windSpeed3pm,
    'Humidity9am': humidity9am,
    'Humidity3pm': humidity3pm,
    'Pressure9am': pressure9am,
    'Pressure3pm': pressure3pm,
    'Cloud9am': cloud9am,
    'Cloud3pm': cloud3pm,
    'Temp9am': temp9am,
    'Temp3pm': temp3pm,
    'RainToday': rainToday
}

def predict():
    model_dict = joblib.load('model/aussie_rain.joblib')
    X_new = preprocess_new_data(input_data, model_dict)
    model = model_dict['model']
    pred = model.predict(X_new)[0]
    proba = model.predict_proba(X_new)[0][1]

    will_rain = 'Yes' if pred == 'Yes' or pred == 1 else 'No'

    st.subheader('Prediction:')
    st.write(f'Will it rain tomorrow? - {will_rain}')
    st.write(f'Probability of rain: {proba:.2%}')


if st.button('Predict'):
    predict()
