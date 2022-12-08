import os
import modal
    
LOCAL = False

if LOCAL == False:
   stub = modal.Stub()
   image = modal.Image.debian_slim().pip_install(["hopsworks","joblib","seaborn","sklearn","dataframe-image"]) 

   @stub.function(image=image, schedule=modal.Period(days=1), secret=modal.Secret.from_name("HOPSWORKS_API_KEY"))
   def f():
       g()


def g():
    import hopsworks
    import pandas as pd
    import numpy as np

    source_url = "https://raw.githubusercontent.com/ID2223KTH/id2223kth.github.io/master/assignments/lab1/titanic.csv"
    project = hopsworks.login()
    fs = project.get_feature_store()
    df = pd.read_csv(source_url)
    
    # filling null values using fillna
    # and dropping unwanted columns
    df.reset_index(drop=True)
    df['Age'].fillna(value=round(df.Age.mean()), inplace=True)
    df['Sex'].replace('male', 0, inplace=True)
    df['Sex'].replace('female', 1, inplace=True)
    df['Sex'] = df['Sex'].astype('str').astype('category')
    df['SibSp'] = df['SibSp'].astype('str').astype('category')
    df['Pclass'] = df['Pclass'].astype('str').astype('category')
    df['Parch'] = df['Parch'].astype('str').astype('category')
    df.drop('Ticket', axis=1, inplace=True)
    df.drop('Name', axis=1, inplace=True)
    df.drop('Embarked', axis=1, inplace=True)
    df.drop('Cabin', axis=1, inplace=True)

    # Aggregating price data into bins:
    bins = [-1, 10, 30, 50, 100, np.inf]
    labels = [1, 2, 3, 4, 5]
    df['Pricerange'] = pd.cut(df['Fare'], bins, labels=labels).astype('str').astype('category')
    df.drop('Fare', axis=1, inplace=True)
    df.drop('PassengerId', axis=1, inplace=True)

    # Reordering columns + ensuring lowercase
    temp_cols=df.columns.tolist()
    new_cols=temp_cols[1:] + temp_cols[0:1]
    df = df[new_cols]
    df.columns = df.columns.str.lower()
    print(df.head())
    
    # Inserting df to feature store
    titanic_fg = fs.get_or_create_feature_group(
        name="titanic_modal_v2",
        version=1,
        primary_key=["pclass","sex","age","sibsp", "parch", "pricerange"],
        description="titanic dataset")
    titanic_fg.insert(df, write_options={"wait_for_job" : False})

if __name__ == "__main__":
    if LOCAL == True :
        g()
    else:
        with stub.run():
            f()