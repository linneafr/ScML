import os
import modal

LOCAL = False

if LOCAL == False:
   stub = modal.Stub()
   image = modal.Image.debian_slim().apt_install(["libgomp1"]).pip_install(["hopsworks", "seaborn", "joblib","dataframe-image", "sklearn"])

   @stub.function(image=image, schedule=modal.Period(days=1), secret=modal.Secret.from_name("HOPSWORKS_API_KEY"))
   def f():
       g()


def generate_passenger(pclass_min, pclass_max, sex, age_min, age_max, sibsp_min, sibsp_max, parch_min,
                       parch_max, pricerange_min, pricerange_max, survived):
    """
    Returns a single passenger as a single row in a DataFrame
    """
    import pandas as pd
    import random

    df = pd.DataFrame({"pclass": [str(random.randint(pclass_min, pclass_max))],
                       "sex": [str(sex)],
                       "age": [random.uniform(age_min, age_max)],
                       "sibsp": [str(random.randint(sibsp_min, sibsp_max))],
                       "parch": [str(random.randint(parch_min, parch_max))],
                       "pricerange": [str(random.randint(pricerange_min, pricerange_max))]
                      })
    df['survived'] = survived
    return df


def get_random_passenger():
    """
    Returns a DataFrame containing one random passenger
    """
    import random

    # randomly pick one type of passenger
    # and write it to the featurestore
    pick_random = random.randint(0, 1)
    if pick_random == 1:
        df = generate_passenger(1, 2, 1, 0, 80, 0, 8, 0, 6, 2, 5, 1)
        print("Survivor added")
    else:
        df = generate_passenger(2, 3, 0, 0, 80, 0, 8, 0, 6, 1, 2, 0)
        print("Non survivor added")

    return df


def g():
    import hopsworks

    project = hopsworks.login()
    fs = project.get_feature_store()

    df = get_random_passenger()

    fg = fs.get_feature_group(name="titanic_modal_v2",version=1)
    fg.insert(df, write_options={"wait_for_job" : False})

if __name__ == "__main__":
    if LOCAL == True :
        g()
    else:
        stub.deploy("titanic-feature-pipeline-daily")
        with stub.run():
            f()
