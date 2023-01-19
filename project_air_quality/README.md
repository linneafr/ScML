# ID2223 - Project : Air Quality Index forecasting service - Reykjavík
Teo Jansson Minne, Linnéa Fredriksson

## The service
This application forecasts the air quality in Reykjavik for the upcoming 7 days based on predictions from weather forecasting data. 
The UI can be accessed through the following Hugging Face space.

https://huggingface.co/spaces/TeoJM/id2223_project_air_quality_weekly_forecast_Reykjavik

<img width="1246" alt="Skärmavbild 2023-01-19 kl  13 01 27" src="https://user-images.githubusercontent.com/26428378/213437905-89d78d7e-5f12-4e41-949a-bdce06404b80.png">


## The data
The historical air quality data was downloaded from [World Air Quality Index](https://aqicn.org/city/iceland/grensasv) from the site Grensásv in Reykjavik.
The data was parsed to the apropriate date-time format and the total AQI was calculated and added to the dataframe. 
The weather data which was downloaded from [VisualCrossing](https://www.visualcrossing.com) was cleaned, and unnecesary data for the task at hand was removed.
This was done in clean_data.ipynb. 

A feature group for [weather data](https://c.app.hopsworks.ai/p/5380/fs/5287/fg/14746) and [AQI data](https://c.app.hopsworks.ai/p/5380/fs/5287/fg/14745) were created and stored at Hopsworks. 
A backfill feature pipeline was created and a daily job scheduled for updating the feature store with the most recent data. 

## The model

In model_selection_training.ipynb the feature groups for the weather and AQI data were joined together to create a joint feature group to be able to perform model training. Four different models were trained for 1-day prediction using xgboost, gradient boosting, random forest and decision tree. 

In order to make the predictions, weather forecasting data for the next 7 days were downloaded directly from VisualCrossing into a data frame and fed into the model. Based on the weather forecasting, a prediction of the corresponding total AQI is made for the following week. 
