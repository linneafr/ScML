# -- coding: utf-8 --
import pandas as pd
from datetime import datetime
import time
import requests
import urllib
from functions import *

import modal

LOCAL=False

if LOCAL == False:
   stub = modal.Stub()
   image = modal.Image.debian_slim().pip_install(["hopsworks==3.0.4"])

   @stub.function(image=image, schedule=modal.Period(days=1), secret=modal.Secret.from_name("hopsworks"))
   def f():
       g()



def g():
    import hopsworks
    date_today = datetime.now().strftime("%Y-%m-%d")
    city = "Reykjavik"
    city_aq = '@8344'

    data_air_quality = [get_air_quality_data(city_aq)]
    data_weather = [get_weather_data(city, date_today)]

    df_air_quality = get_air_quality_df(data_air_quality)
    print(df_air_quality)
    df_weather = get_weather_df(data_weather)
    print(df_weather)

    project = hopsworks.login()

    fs = project.get_feature_store()

    air_quality_fg = fs.get_or_create_feature_group(
        name = 'air_quality_fg',
        version = 1
    )
    weather_fg = fs.get_or_create_feature_group(
        name = 'weather_fg',
        version = 1
    )
    air_quality_fg.insert(df_air_quality)
    weather_fg.insert(df_weather)


if __name__ == "__main__":
    if LOCAL == True :
        g()
    else:
        with stub.run():
            f()
