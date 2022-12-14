import os
import modal

LOCAL = False

if LOCAL == False:
    stub = modal.Stub()
    image = modal.Image.debian_slim().apt_install(["libgomp1"]).pip_install(["hopsworks", "seaborn", "joblib","dataframe-image", "scikit-learn", "xgboost"])

   
    @stub.function(image=image, schedule=modal.Period(days=1), secret=modal.Secret.from_name("HOPSWORKS_API_KEY"))
    def f():
        g()

def g():
    # import datetime
    from datetime import datetime
    import dataframe_image as dfi
    import hopsworks
    import joblib
    import pandas as pd
    import requests
    import seaborn as sns
    from PIL import Image
    from sklearn.metrics import confusion_matrix

    survivor_url = "https://huggingface.co/spaces/TeoJM/Titanic/resolve/main/images/survived.png"
    died_url = "https://huggingface.co/spaces/TeoJM/Titanic/resolve/main/images/died.png"
    project = hopsworks.login()
    fs = project.get_feature_store()
    
    mr = project.get_model_registry()
    model = mr.get_model("titanic_modal_v2", version=1)
    model_dir = model.download()
    model = joblib.load(model_dir + "/titanic_model.pkl")
    
    feature_view = fs.get_feature_view(name="titanic_modal_v2", version=1)
    batch_data = feature_view.get_batch_data()

    # Change data type for compatability with xgboost
    batch_data['pclass'] = batch_data['pclass'].astype(int)
    batch_data['sex'] = batch_data['sex'].astype(int)
    batch_data['sibsp'] = batch_data['sibsp'].astype(int)
    batch_data['parch'] = batch_data['parch'].astype(int)
    batch_data['pricerange'] = batch_data['pricerange'].astype(int)
    # batch_data['survived'] = batch_data['survived'].astype(int)
    
    y_pred = model.predict(batch_data)
    
    offset = 1
    survived = y_pred[y_pred.size-offset]
    print("\nPredicted survivor status: "+str(survived))
    dataset_api = project.get_dataset_api()    
    
    fg = fs.get_feature_group(name="titanic_modal_v2", version=1)
    df = fg.read()
    label = df.iloc[-offset]["survived"]
    print("Actual survivor status: "+str(label))
    
    monitor_fg = fs.get_or_create_feature_group(name="titanic_predictions",
                                                version=1,
                                                primary_key=["datetime"],
                                                description="titanic survival Prediction/Outcome Monitoring"
                                                )
    
    now = datetime.now().strftime("%m/%d/%Y, %H:%M:%S")
    data = {
        'prediction': [survived],
        'label': [label],
        'datetime': [now],
       }
    monitor_df = pd.DataFrame(data)
    monitor_fg.insert(monitor_df, write_options={"wait_for_job" : False})
    
    history_df = monitor_fg.read()
    # Add our prediction to the history, as the history_df won't have it - 
    # the insertion was done asynchronously, so it will take ~1 min to land on App
    history_df = pd.concat([history_df, monitor_df])

    df_recent = history_df.tail(5)
    dfi.export(df_recent, './df_recent_titanic.png', table_conversion = 'matplotlib')
    dataset_api.upload("./df_recent_titanic.png", "Resources/images", overwrite=True)
    
    # Set prediction image
    if survived == 1:
        pred_img = Image.open(requests.get(survivor_url, stream=True).raw)
    else:
        pred_img = Image.open(requests.get(died_url, stream=True).raw)
    # Set label image
    if survived == 1:
        label_img = Image.open(requests.get(survivor_url, stream=True).raw)
    else:
        label_img = Image.open(requests.get(died_url, stream=True).raw)
        
    pred_img.save("./latest_pred.png")
    label_img.save("./latest_label.png")
    
    dataset_api.upload("./latest_pred.png", "Resources/images", overwrite=True)
    dataset_api.upload("./latest_label.png", "Resources/images", overwrite=True)
    
    predictions = history_df[['prediction']].astype('int')
    labels = history_df[['label']].astype('int')

    # Only create the confusion matrix when our titanic_predictions feature group has examples of all 2 titanic predictions
    print("Number of different Titanic passenger predictions to date: " + str(predictions.value_counts().count()))
    if predictions.value_counts().count() == 2:
        results = confusion_matrix(labels, predictions)
    
        df_cm = pd.DataFrame(results, ['True Non-Survivor', 'True Survivor'],
                         ['Pred Non-Survivor', 'Pred Survivor'])
    
        cm = sns.heatmap(df_cm, annot=True)
        fig = cm.get_figure()
        fig.savefig("./confusion_matrix_titanic.png")
        dataset_api.upload("./confusion_matrix_titanic.png", "Resources/images", overwrite=True)
    else:
        print("You need 2 different titanic predictions to create the confusion matrix.")
        print("Run the batch inference pipeline more times until you get 2 different titanic predictions") 

if __name__ == "__main__":
    if LOCAL == True :
        g()
    else:
        stub.deploy("titanic-batch-inference-pipeline")
        # with stub.run():
        #     f()