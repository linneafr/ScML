import hopsworks
import os
import pandas as pd
import datetime
import time
import requests
import joblib
import numpy as np
from hsml.engine.model_engine import RestAPIError


def get_air_quality_data(city_name, AIR_QUALITY_API_KEY):
    json = get_air_json(city_name, AIR_QUALITY_API_KEY)
    iaqi = json['iaqi']
    forecast = json['forecast']['daily']
    
    params = ['pm10', 'pm25', 'no2', 'so2']
    for param in params:
        if param not in iaqi:
            iaqi[param] = {"v": np.nan}
    
    return [      
        iaqi['pm25']['v'],
        iaqi['pm10']['v'],
        iaqi['no2']['v'],
        iaqi['so2']['v'],
        json['aqi']
    ]

def get_air_json(city_name, AIR_QUALITY_API_KEY):
    return requests.get(f'https://api.waqi.info/feed/{city_name}/?token={AIR_QUALITY_API_KEY}').json()['data']

def timestamp_2_time(x):
    dt_obj = datetime.datetime.strptime(str(x), '%Y-%m-%d')
    dt_obj = dt_obj.timestamp() * 1000
    return int(dt_obj)

def get_weather_data(city, curr_date, n_days, WEATHER_API_KEY):
    end_date = (curr_date + datetime.timedelta(days=n_days-1)).strftime("%Y-%m-%d")
    start_date = curr_date.strftime("%Y-%m-%d")

    daily_data_json = requests.get(f'https://weather.visualcrossing.com/VisualCrossingWebServices/rest/services/timeline/{city}/{start_date}/{end_date}?unitGroup=metric&include=days&key={WEATHER_API_KEY}&contentType=json').json()
    daily_data = daily_data_json['days']
    df = pd.DataFrame()
    for i in range(n_days):
        data = daily_data[i]
        data_to_df = np.array([
        daily_data_json['address'].capitalize(),
        data['datetime'],
        data['tempmax'],
        data['tempmin'],
        data['temp'],
        data['feelslikemax'],
        data['feelslikemin'],
        data['feelslike'],
        data['dew'],
        data['humidity'],
        data['precip'],
        data['precipprob'],
        data['precipcover'],
        data['preciptype'],
        data['snow'],
        data['snowdepth'],
        data['windgust'],
        data['windspeed'],
        data['winddir'],
        data['pressure'],
        data['cloudcover'],
        data['visibility'],
        data['solarradiation'],
        data['solarenergy'],
        data['uvindex'],
        data['conditions']
        ]).reshape((-1,26))
        df = pd.concat([df, get_weather_df(data_to_df)])
    df = df.reset_index().drop(columns=['index'])
    return df

def get_weather_df(data):
    col_names = [
        'address',
        'date',
        'tempmax',
        'tempmin',
        'temp',
        'feelslikemax',
        'feelslikemin',
        'feelslike',
        'dew',
        'humidity',
        'precip',
        'precipprob',
        'precipcover',
        'preciptype',
        'snow',
        'snowdepth',
        'windgust',
        'windspeed',
        'winddir',
        'pressure',
        'cloudcover',
        'visibility',
        'solarradiation',
        'solarenergy',
        'uvindex',
        'severerisk'
    ]

    new_data = pd.DataFrame(
        data,
        columns=col_names
    )
    new_data.date = new_data.date.apply(timestamp_2_time)

    return new_data

def daily_to_weekly(df):
    df_temp = pd.DataFrame()
    for column in df.columns:
      for i in range(0,7):
          df_temp[f'{column}_{i}'] = df[column].shift(-i)
    return df_temp.iloc[0].values.astype(float)


if __name__ == "__main__":
    model_name = "gb_base_v2"
    
    
    date_today = datetime.datetime.now()
    city_weather = "Reyjkjavik"
    city_aq = '@8344'
    

    df = get_weather_data(city_weather, date_today, 7, os.getenv('WEATHER_API_KEY'))

    df['preciptype'] = df['preciptype'].astype(str)
    df['preciptype'].replace("['rain']", 1, inplace=True)
    df['preciptype'].replace("['rain', 'snow']", 2, inplace=True)
    df['preciptype'].replace("['snow']", 3, inplace=True)
    df['preciptype'].replace("['rain', 'freezingrain', 'snow']", 4, inplace=True)
    df['precipprob'].replace(100, 1, inplace=True)
    df = df.fillna(0)
    weather_cols = ['tempmax', 'tempmin', 'temp',
        'dew', 'precip', 'precipprob', 'preciptype', 'snowdepth', 'windspeed',
        'winddir', 'visibility', 'solarradiation']
    df2 = df[weather_cols]

    weather_data = daily_to_weekly(df2)
    airq_data = get_air_quality_data(city_aq, os.getenv('AIR_QUALITY_API_KEY'))
    input = np.concatenate((airq_data, weather_data))

    scX = joblib.load('/models/standardscalerX.bin')
    scy = joblib.load('/models/standardscalery.bin')

    input_t = scX.transform(input.reshape(1, -1))


    project = hopsworks.login(project="linneafr", api_key_value=os.getenv('HOPSWORKS_API_KEY'))

    mr = project.get_model_registry()

    try:
        model = mr.get_model(model_name, version=1)
        model_dir = model.download()
        model = joblib.load(model_dir + "/model.pkl")
    except RestAPIError:
        model = joblib.load(f"/models/{model_name}.pkl")
        
    output = scy.inverse_transform(model.predict(input_t).reshape(-1,7)).squeeze().astype('int')
    print(output)