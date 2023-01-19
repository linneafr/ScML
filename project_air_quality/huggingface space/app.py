import hopsworks
import pandas as pd
import datetime
import time
import requests
import joblib
import numpy as np
import gradio as gr
import sklearn
from hsml.engine.model_engine import RestAPIError

AIR_QUALITY_API_KEY = "59698ca239fd8be4728d0ff939a251195f16bec5"
WEATHER_API_KEY = "NSDUNZJ59YSSHWMFZF9LCADY5"
HOPSWORKS_API_KEY = "jJP87Tuw2eB9hmIQ.2yOFcHCfIUSSmnZTlsqOzbDw90syY93cQUAuyqGdzn1wwubawaIXc9mOYQ1wqjnW"

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
    return requests.get('https://api.waqi.info/feed/'+city_name+'/?token='+AIR_QUALITY_API_KEY).json()['data']

def timestamp_2_time(x):
    dt_obj = datetime.datetime.strptime(str(x), '%Y-%m-%d')
    dt_obj = dt_obj.timestamp() * 1000
    return int(dt_obj)
    
def replace_none_with_zero(some_dict):
            return { k: (0 if v is None else v) for k, v in some_dict.items() }
    
def get_weather_data(city, curr_date, n_days, WEATHER_API_KEY):
    end_date = (curr_date + datetime.timedelta(days=n_days-1)).strftime("%Y-%m-%d")
    start_date = curr_date.strftime("%Y-%m-%d")

    daily_data_json = requests.get('https://weather.visualcrossing.com/VisualCrossingWebServices/rest/services/timeline/'+city+'/'+start_date+'/'+end_date+'?unitGroup=metric&include=days&key='+WEATHER_API_KEY+'&contentType=json').json()
    daily_data = daily_data_json['days']
    print(daily_data)
    df = pd.DataFrame()
    for i in range(n_days):
        data = daily_data[i]
        data = replace_none_with_zero(data)
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
            # data['preciptype'],
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
            ]).reshape((-1,25))
        print(df)
        print(data_to_df)
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
        # 'preciptype',
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

def get_data():
    date_today = datetime.datetime.now()
    city_weather = "Reyjkjavik"
    city_aq = '@8344'
    
    df = get_weather_data(city_weather, date_today, 7, "NSDUNZJ59YSSHWMFZF9LCADY5")
    
    # df['preciptype'] = df['preciptype'].astype(str)
    # df['preciptype'].replace("", 0, inplace=True)
    # df['preciptype'].replace("['rain']", 1, inplace=True)
    # df['preciptype'].replace("['rain', 'snow']", 2, inplace=True)
    # df['preciptype'].replace("['snow']", 3, inplace=True)
    # df['preciptype'].replace("['rain', 'freezingrain', 'snow']", 4, inplace=True)
    df['precipprob'].replace(100, 1, inplace=True)
    df = df.fillna(0)
    weather_cols = ['tempmax', 'tempmin', 'temp',
        'dew', 'precip', 'precipprob', 'snowdepth', 'windspeed',
        'winddir', 'visibility', 'solarradiation']
    df2 = df[weather_cols]
    
    weather_data = daily_to_weekly(df2)
    airq_data = get_air_quality_data(city_aq, AIR_QUALITY_API_KEY)
    aqi = airq_data[0]
    input = np.concatenate((airq_data, weather_data))
    return input, aqi, date_today
    
def get_model(model_name):
    project = hopsworks.login(project="linneafr", api_key_value=HOPSWORKS_API_KEY)
    mr = project.get_model_registry()
    
    try:
        model = mr.get_model(model_name, version=1)
        model_dir = model.download()
        model = joblib.load(model_dir + "/"+model_name+".pkl")
    except RestAPIError:
        model = joblib.load("/models/"+model_name+".pkl")
        
    return model
    
def get_forecast(model_name):
    input, aqi, date_today = get_data()
    scX = joblib.load('standardscalerX.save')
    scy = joblib.load('standardscalery.save')
    input_t = scX.transform(input.reshape(1, -1))
    
    if model_name == "ensemble (recommended)":
        outputs = np.zeros((len(model_names),7))
        for model_i, model_name in enumerate(model_names[1:]):
            model = get_model(model_name)
            outputs[model_i] = scy.inverse_transform(model.predict(input_t).reshape(-1,7)).squeeze().astype('int')
            del model
        output = outputs.mean(axis=0)
    else:
        model = get_model(model_name)
        output = scy.inverse_transform(model.predict(input_t).reshape(-1,7)).squeeze().astype('int')
        del model
    dates = np.array([(date_today + datetime.timedelta(days=days)).strftime("%Y-%m-%d") for days in range(8)])
    output = np.concatenate((np.array([aqi]),output))
    output_table = pd.DataFrame({'Date':dates, 'AQI (predicted)' : output})
    
    return output_table

model_names = ['ensemble (recommended)','rf_base', 'rf_large', 'gb_base_v3', 'gb_large_v3', 'gb_small_v3', 'gb_tiny_v3']


with gr.Blocks() as demo:
    gr.Label("Predict air quality in Reykjavik, Iceland")
    gr.Image("iceland_picture.jpg")
    with gr.Row():
        inp = gr.Radio(model_names, value="ensemble (recommended)")
        out = gr.DataFrame()
    btn = gr.Button("Run prediction")
    btn.click(fn=get_forecast, inputs=inp, outputs=out)

demo.launch()

# iface = gr.Interface(
#     fn=get_forecast, 
#     inputs=gr.Radio(model_names, value="ensemble (recommended)"),
#     outputs=gr.DataFrame(),
#     title="Run prediction"
# )
# iface.launch()