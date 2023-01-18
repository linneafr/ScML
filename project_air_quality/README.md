# ID2223 - Project : Air Quality Index forecasting service - Reykjavík
Teo Jansson Minne, Linnéa Fredriksson

## The service
This application forecasts the air quality in Reykjavik for the following 7 days based on predictions from weather forecasting data. 
The UI can be accessed through the following Hugging Face space.

https://huggingface.co/spaces/TeoJM/id2223_project_air_quality_weekly_forecast_Reykjavik

## The data
The raw air quality data was downloaded from World Air Quality Index from the site Grensásv in Reykjavik.
The data was parsed to the apropriate date-time format and AQI was calculated and added to the dataframe. 
The weather data which was downloaded from VisualCrossing was cleaned, and unnecesary data for the task at hand was removed.

The AQI data and weather data was joined together to create a feature group at hopsworks https://c.app.hopsworks.ai/p/5380/fs/5287/fg/14750 
A backfill feature pipeline was created and a weekly job scheduled for updating the model. 

## The model
