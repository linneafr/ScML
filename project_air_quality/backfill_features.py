import pandas as pd
from functions import *
import hopsworks

# load the data:

df_air_quality = pd.read_csv('air_quality_data.csv')
df_weather = pd.read_csv('weather_data.csv')
df_more_weather = pd.read_csv('more_weather_data.csv')

def timestamp_2_time(x):
    dt_obj = datetime.strptime(str(x), '%Y-%m-%d')
    dt_obj = dt_obj.timestamp() * 1000
    return int(dt_obj)

df_air_quality.date = df_air_quality.date.apply(timestamp_2_time)
df_weather.datetime = df_weather.datetime.apply(timestamp_2_time)
df_more_weather.datetime = df_more_weather.datetime.apply(timestamp_2_time)

project = hopsworks.login()
fs = project.get_feature_store()

# Creating air quality feature groups:
air_quality_fg = fs.get_or_create_feature_group(
        name = 'air_quality_fg',
        description = 'Air Quality characteristics of each day',
        version = 1,
        primary_key = ['date'],
        online_enabled = True,
        event_time = 'date'
    )

air_quality_fg.insert(df_air_quality)

# Creating weather data feature groups:
weather_fg = fs.get_or_create_feature_group(
        name = 'weather_fg',
        description = 'Weather characteristics of each day',
        version = 1,
        primary_key = ['datetime'],
        online_enabled = True,
        event_time = 'datetime'
    )

weather_fg.insert(df_weather)
weather_fg.insert(df_more_weather)
