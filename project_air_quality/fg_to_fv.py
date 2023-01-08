import hopsworks
import pandas as pd

def main():
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

    # Get feature groups
    weather_df = weather_fg.select_all().read()
    weather_df.rename(columns={'datetime':'date'}, inplace=True)
    air_quality_df = air_quality_fg.select_all().read()

    # Join dataframes and clean data
    joint_df = pd.merge(air_quality_df, weather_df)
    joint_df = joint_df.sort_values(by=['date'], ignore_index=True).drop(['address', 'co'], axis=1)
    joint_df['precipprob'].replace(100, 1, inplace=True)
    joint_df = joint_df.fillna(0)

    # Create joint feature group
    air_weather_data_joint_fg = fs.get_or_create_feature_group(
            name = 'air_weather_data_joint_fg',
            description = 'Air Quality and Weather characteristics of each day',
            version = 1,
            primary_key = ['date'],
            online_enabled = True,
            event_time = 'date'
        )

    air_weather_data_joint_fg.insert(joint_df)

    # Create joint feature view
    identity_columns = ['date', 'aqi'] # Index and output is not transformed
    mapping_transformers = {col_name:fs.get_transformation_function(name='standard_scaler') for col_name in joint_df.columns if col_name not in identity_columns}
    query = air_weather_data_joint_fg.select_all()

    feature_view = fs.create_feature_view(
        name = 'air_weather_data_joint_fv',
        version = 1,
        transformation_functions = mapping_transformers,
        query = query
    )
    
if __name__ == '__main__':
    main()