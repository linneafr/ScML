import os
import modal

LOCAL = False

if LOCAL == False:
   stub = modal.Stub()
   image = modal.Image.debian_slim().apt_install(["libgomp1"]).pip_install(["hopsworks", "seaborn", "joblib","dataframe-image", "sklearn"])

   @stub.function(image=image, schedule=modal.Period(days=1), secret=modal.Secret.from_name("HOPSWORKS_API_KEY"))
   def f():
       g()


def generate_passenger(train_set):
    """
    Returns a single passenger data point as a single row in a DataFrame
    "Cheats" to get reasonable values by randomly sampling each category
    from the training set
    """
    import pandas as pd
    import numpy as np

    df = pd.DataFrame({"pclass": [np.random.choice(train_set['pclass'])],
                       "sex": [np.random.choice(train_set['sex'])],
                       "age": [np.random.choice(train_set['age'])],
                       "sibsp": [np.random.choice(train_set['sibsp'])],
                       "parch":[np.random.choice(train_set['parch'])],
                       "pricerange":[np.random.choice(train_set['pricerange'])],
                       'survived':[np.random.choice(np.array([0,1]))]
                      })
    return df

def g():
    import hopsworks

    project = hopsworks.login()
    fs = project.get_feature_store()
    feature_view = fs.get_feature_view(name="titanic_modal_v2",
                                        version=1)  
    train_set, _, _, _ = feature_view.train_test_split(0.2)
    df = generate_passenger(train_set)

    fg = fs.get_feature_group(name="titanic_modal_v2",version=1)
    fg.insert(df, write_options={"wait_for_job" : False})

if __name__ == "__main__":
    if LOCAL == True :
        g()
    else:
        with stub.run():
            f()