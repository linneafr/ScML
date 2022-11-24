import os
import modal

LOCAL=True

if LOCAL == False:
   stub = modal.Stub()
   image = modal.Image.debian_slim().pip_install(["hopsworks==3.0.4"])

   @stub.function(image=image, schedule=modal.Period(days=1), secret=modal.Secret.from_name("HOPSWORKS_API_KEY"))
   def f():
       g()


def generate_passenger(pclass_max, pclass_min, sex, age_max, age_min, sibsp_max, sibsp_min, parch_max,
                       parch_min, pricerange_max, pricerange_min, survived):
    """
    Returns a single passenger as a single row in a DataFrame
    """
    import pandas as pd
    import random

    df = pd.DataFrame({"pclass": [random.randint(pclass_min, pclass_max)],
                       "sex": [sex],
                       "age": [random.randint(age_min, age_max)],
                       "sibsp": [random.randint(sibsp_min, sibsp_max)],
                       "parch": [random.randint(parch_min, parch_max)],
                       "pricerange": [random.randint(pricerange_min, pricerange_max)]
                      })
    df['survived'] = survived
    return df


def get_random_passenger():
    """
    Returns a DataFrame containing one random passenger
    """
    import pandas as pd
    import random

    #Surviving passenger
    survivor_df = generate_passenger(1, 2, 1, 0, 80, 0, 8, 0, 6, 2, 5, 1)
    #Non surviving passenger
    non_survivor_df = generate_passenger(2, 3, 0, 0, 80, 0, 8, 0, 6, 1, 2, 0)


    # randomly pick one of these and write it to the featurestore
    pick_random = random.randint(0, 1)
    if pick_random == 1:
        titanic_df = survivor_df
        print("Survivor added")
    else:
        titanic_df = non_survivor_df
        print("Non survivor added")

    return titanic_df


def g():
    import hopsworks

    project = hopsworks.login()
    fs = project.get_feature_store()

    titanic_df = get_random_passenger()

    titanic_fg = fs.get_feature_group(name="titanic_modal",version=1)
    titanic_fg.insert(titanic_df, write_options={"wait_for_job" : False})

if __name__ == "__main__":
    if LOCAL == True :
        g()
    else:
        with stub.run():
            f()
