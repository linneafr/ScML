import os
import modal

LOCAL = False

if LOCAL == False:
   stub = modal.Stub()
   image = modal.Image.debian_slim().apt_install(["libgomp1"]).pip_install(["hopsworks", "seaborn", "joblib", "scikit-learn", "xgboost"])

   @stub.function(image=image, schedule=modal.Period(days=1), secret=modal.Secret.from_name("HOPSWORKS_API_KEY"))
   def f():
       g()


def g():
    import hopsworks
    import pandas as pd
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.metrics import accuracy_score
    from sklearn.metrics import confusion_matrix
    from sklearn.metrics import classification_report
    import seaborn as sns
    from matplotlib import pyplot
    from hsml.schema import Schema
    from hsml.model_schema import ModelSchema
    import joblib
    import xgboost as xgb

    project = hopsworks.login()
    fs = project.get_feature_store()
    try:
        feature_view = fs.get_feature_view(name="titanic_modal_v2", version=1)
    except:
        titanic_fg = fs.get_feature_group(name="titanic_modal_v2", version=1)
        query = titanic_fg.select_all()
        feature_view = fs.create_feature_view(name="titanic_modal_v2",
                                          version=1,
                                          description="Read from Titanic passenger dataset",
                                          labels=["survived"],
                                          query=query)

    # Read and split train/test data
    X_train, X_test, y_train, y_test = feature_view.train_test_split(0.2)

    # Change data type for compatability with xgboost
    X_train['pclass'] = X_train['pclass'].astype(int)
    X_train['sex'] = X_train['sex'].astype(int)
    X_train['sibsp'] = X_train['sibsp'].astype(int)
    X_train['parch'] = X_train['parch'].astype(int)
    X_train['pricerange'] = X_train['pricerange'].astype(int)
    y_train['survived'] = y_train['survived'].astype(int)
    
    X_test['pclass'] = X_test['pclass'].astype(int)
    X_test['sex'] = X_test['sex'].astype(int)
    X_test['sibsp'] = X_test['sibsp'].astype(int)
    X_test['parch'] = X_test['parch'].astype(int)
    X_test['pricerange'] = X_test['pricerange'].astype(int)
    y_test['survived'] = y_test['survived'].astype(int)

    # Train our model with the Scikit-learn K-nearest-neighbors algorithm using our features (X_train) and labels (y_train)
    # model = KNeighborsClassifier(n_neighbors=2)
    # model.fit(X_train, y_train.values.ravel())
    
    # Train our model with xgboost classifier
    model = xgb.XGBClassifier()
    model.fit(X_train, y_train.values.ravel())

    # Evaluate model performance using the features from the test set (X_test)
    y_pred = model.predict(X_test)

    # Compare predictions (y_pred) with the labels in the test set (y_test)
    metrics = classification_report(y_test, y_pred, output_dict=True)
    results = confusion_matrix(y_test, y_pred)

    # Create the confusion matrix as a figure, we will later store it as a PNG image file
    df_cm = pd.DataFrame(results, ['True Non-Survivor', 'True Survivor'],
                         ['Pred Non-Survivor', 'Pred Survivor'])
    cm = sns.heatmap(df_cm, annot=True)
    fig = cm.get_figure()

    # We will now upload our model to the Hopsworks Model Registry. First get an object for the model registry.
    mr = project.get_model_registry()

    # The contents of the 'titanic_model' directory will be saved to the model registry. Create the dir, first.
    model_dir="titanic_model"
    if os.path.isdir(model_dir) == False:
        os.mkdir(model_dir)

    # Save both our model and the confusion matrix to 'model_dir', whose contents will be uploaded to the model registry
    joblib.dump(model, model_dir + "/titanic_model.pkl")
    fig.savefig(model_dir + "/confusion_matrix.png")


    # Specify the schema of the model's input/output using the features (X_train) and labels (y_train)
    input_schema = Schema(X_train)
    output_schema = Schema(y_train)
    model_schema = ModelSchema(input_schema, output_schema)

    # Create an entry in the model registry that includes the model's name, desc, metrics
    titanic_model = mr.python.create_model(
        name="titanic_modal_v2",
        metrics={"accuracy" : metrics['accuracy']},
        model_schema=model_schema,
        description="Titanic Dataset Predictor"
    )

    # Upload the model to the model registry, including all files in 'model_dir'
    titanic_model.save(model_dir)

if __name__ == "__main__":
    if LOCAL == True :
        g()
    else:
        with stub.run():
            f()
