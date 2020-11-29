import streamlit as st
# To make things easier later, we're also importing numpy and pandas for
# working with sample data.
import numpy as np
import pandas as pd
import seaborn as sns
from joblib import load
import matplotlib.pyplot as plt
from datetime import datetime
import sklearn
import pickle
from sklearn.preprocessing import StandardScaler

sns.set_style('darkgrid')

# DATA SOURCES
FILTERED_CALI_DATA_WITH_COUNTY = './Data/USW00003171.csv'
WEATHER_DATA_URL = './Data/dfWeatherAndWildfire.csv'
WILDFIRE_DATA_URL = './Data/CaliDataWithCounty.csv'

# For loading data from these sources


@st.cache
def load_data(nrows, source):
    return pd.read_csv(source, nrows=nrows)


option = st.sidebar.selectbox(
    'Options',
    [
        'Wildfire Overview',
        'Exploratory Data Analysis',
        'Predict Wildfire'
    ])

'You selected:', option


def introduction_func():
    st.title('Wildfire Prediction')
    """
    # Introduction to Data Sources
    We will be importing two datasets here. The first one represents the Dataset that is collected from various resources and the weather data set is downloaded from the **RIVERSIDE MUNICIPAL AIRPORT, CA US** Weather Center(This station can be found on [NOAA Website](https://www.ncdc.noaa.gov/cdo-web/datasets/GHCND/stations/GHCND:USW00003171/detail) ). The data available in these weather station range from April,1998 to current date. 

    Considering **Riverside County** for our analysis is based on the data and visualizations that we see below. Lets go through the various data sources.
    """
    # Create a text element and let the reader know the data is loading.
    data_load_state = st.text('Loading data...')

    # Load 100 rows of Wildfire data into the dataframe.
    data = load_data(100, WILDFIRE_DATA_URL)
    st.subheader('Raw Wildfire Data')
    st.write(data)

    # Load 100 rows of Wildfire data into the dataframe.
    data = load_data(100, FILTERED_CALI_DATA_WITH_COUNTY)
    st.subheader('Raw Weather Data')
    st.write(data)

    # Notify the reader that the data was successfully loaded.
    data_load_state.text('Loading data...done!')

    """
    Data was extracted from the above two data sets from the year 1998 to year 2015 and Joined to form
    a consolidated dataset. This is available in the Exploratory Data Analysis Section.
    """

    # Create a text element and let the reader know the data is loading.
    data_load_state = st.text('Loading data...')

    # Load 100 rows of Weather data into the dataframe.
    data = load_data(100, WEATHER_DATA_URL)
    st.subheader('Raw Consolidated Data')
    st.write(data)

    # Notify the reader that the data was successfully loaded.
    data_load_state.text('Loading data...done!')


def exploratory_analysis():
    # Data import for California Wildfire Data
    dfCaliData = pd.read_csv(WILDFIRE_DATA_URL)

    st.title('Exploratory Data Analysis for Wildfire')

    # Select the columns required for our analysis leaving others
    dfCaliData = dfCaliData[['DISCOVERY_DATE', 'DISCOVERY_TIME',
                             'STAT_CAUSE_DESCR', 'FIRE_SIZE', 'FIRE_SIZE_CLASS', 'NewCountyValue']]

    # Lets convert the Discovery Date to Proper Date
    dfCaliData['DATE'] = pd.to_datetime(
        dfCaliData['DISCOVERY_DATE'] - pd.Timestamp(0).to_julian_date(), unit='D')

    # Adding Month Column and Day of the week in order to Aid visualization
    dfCaliData['MONTH'] = pd.DatetimeIndex(dfCaliData['DATE']).month
    dfCaliData['DAY_OF_WEEK'] = dfCaliData['DATE'].dt.strftime("%A")

    # Drop the DISCOVERY_DATE column as the DATE column already holds the required data
    dfCaliData = dfCaliData.drop(['DISCOVERY_DATE'], axis=1)

    # Graphical representation of the above data
    col1, col2 = st.beta_columns(2)
    with col1:
        st.markdown(
            f"""
        <style>
            .reportview-container .main .block-container{{
                max-width: {100}%;
                padding-top: {1}rem;
                padding-right: {1}rem;
                padding-left: {1}rem;
                padding-bottom: {1}rem;
            }}
        </style>
        """,
            unsafe_allow_html=True,
        )
        """
        ### Cause of Wildfire in California
        We will check the frequency of the wildfire with the causes that caused it.
        Lets get the frequency count for the Cause of the wild fire
        """
        fig = sns.displot(data=dfCaliData, y="STAT_CAUSE_DESCR", hue="STAT_CAUSE_DESCR",
                          multiple="stack").set(title='Causes of Wildfire across California')
        plt.xlabel('Number of incidents')
        plt.ylabel('Causes of Wildfire Incidents')
        st.pyplot(fig)

        """
        From the above data as well as the Graph it is evident that the major cause of wildfire is **Miscellaneous(unknown)** whereas the second largest cause is **Equipement usage(Caused by human intervention)**. As the list suggests **Lightening**, **Arson** and **Debris Burning** also contribute to the cause.
        """

    with col2:
        """
        ### Most active Months

        The monthly analysis will show us which months are prone to wildfire incidents. These wildfire incidents increase from a period of May to Septemper, which mostly is a dry spell in California. We should also note that these months are the dry months in california.
        """
        # Graphical representation of the above data
        dfMonth = dfCaliData.groupby(['MONTH']).size()
        dfMonth = dfMonth.reset_index(name='count').sort_values('count')

        # Creating barplot
        fig, ax = plt.subplots()
        ax = sns.barplot(data=dfMonth, y='count', x='MONTH').set(
            title='Wildfire per Month')
        plt.xlabel('Month of the year')
        plt.ylabel('Number of Wildfire Incidents')
        st.pyplot(ax)


def predict_wildfire():

    with open('./data/rf.pkl', 'rb') as file:
        print(file)
        model = pickle.load(file)
    
    with open('./data/sc.pkl', 'rb') as file:
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
    if st.button('add'):
        y_predict = model.predict(X_test)
        st.text(y_predict)
        st.text(X_test)


# Various views that should be shown based on the option
if option == 'Wildfire Overview':
    introduction_func()

if option == 'Exploratory Data Analysis':
    exploratory_analysis()

if option == 'Predict Wildfire':
    predict_wildfire()
