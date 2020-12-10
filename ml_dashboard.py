import streamlit as st
# To make things easier later, we're also importing numpy and pandas for
# working with sample data.
import numpy as np
import pandas as pd
from joblib import load
import matplotlib.pyplot as plt
from datetime import datetime
import sklearn
import pickle
from sklearn.preprocessing import StandardScaler


# DATA SOURCES
FILTERED_CALI_DATA_WITH_COUNTY = './Data/USW00003171.csv'
WEATHER_DATA_URL = './Data/dfWeatherAndWildfire.csv'
WILDFIRE_DATA_URL = './Data/CaliDataWithCounty.csv'

# For loading data from these sources


@st.cache
def load_data(nrows, source):
    return pd.read_csv(source, nrows=nrows)


# option = st.sidebar.selectbox(
#     'Options',
#     [
#         'Wildfire Overview',
#         'Exploratory Data Analysis',
#         'Predict Wildfire'
#     ])

# 'You selected:', option

def predict_wildfire():

    with open('./streamlitdata/rf.pkl', 'rb') as file:
        print(file)
        model = pickle.load(file)
    
    with open('./streamlitdata/sc.pkl', 'rb') as file:
        print(file)
        sc = pickle.load(file)

    today = datetime.today()
    dt_string = today.strftime("%H%M.%S")

    month_of_year_dict = {'Jan': 1, 'Feb': 2, 'Mar': 3, 'Apr': 4, 'May': 5,
                          'Jun': 6, 'July': 7, 'Aug': 8, 'Sep': 9, 'Oct': 10, 'Nov': 11, 'Dec': 12}
    days_of_week_dict = {'Mon': 1, 'Tue': 2, 'Wed': 3,
                         'Thu': 4, 'Fri': 5, 'Sat': 6, 'Sun': 0}

    st.title('Predict Wildfire')
    st.write(sklearn.__version__)

    # Slider for Temperature Min
    min_temp, max_temp = st.slider(
        'Select the Min and Max Temperature in Fahrenheit',
        -4.0, 140.0,  (63.0, 85.0))

    # Precipitation Range
    precp = st.slider(
        'Select the Avg. Precipitation on a given day in mm',
        0.0, 280.0)

    # Average Wind Speed on a given Day
    wind_speed = st.slider(
        'Select the Avg. wind speed on a given day in m/sec',
        -4.0, 140.0)

    month = st.select_slider(
        'Select a Month',
        options=['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                 'July', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])

    day_of_week = st.select_slider(
        'Day of the week',
        options=['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'])

    # Values that needs to be parced in order for the model to work ['TMAX','TMIN','TAVG','PRCP','AWND','MONTH','DAY_OF_WEEK','DISCOVERY_TIME']
    X_test = [[max_temp, min_temp, (max_temp+min_temp)/2, precp,
              wind_speed, month_of_year_dict[month], days_of_week_dict[day_of_week], 0]]
    #scaler = StandardScaler()
    #X_test = scaler.fit_transform(X_test)
    X_test = sc.transform(X_test)
    if st.button('Detect Chance of Wildfire'):
        y_predict = model.predict(X_test)
        st.text(y_predict)
        if y_predict:
            st.error('THERE IS A CHANCE OF WILDFIRE OCCURANCE')
        else:
            st.success('THERE WILL BE NO WILDFIRE')

option='Predict Wildfire'
# Various views that should be shown based on the option
if option == 'Predict Wildfire':
    predict_wildfire()
