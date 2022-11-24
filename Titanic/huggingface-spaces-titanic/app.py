import gradio as gr
import numpy as np
import hopsworks
import joblib

project = hopsworks.login()
fs = project.get_feature_store()


mr = project.get_model_registry()
model = mr.get_model("titanic_modal_v2", version=1)
model_dir = model.download()
model = joblib.load(model_dir + "/titanic_model.pkl")


def titanic(pclass, sex, age, sibsp, parch, pricerange):
    input_list = []
    input_list.append(pclass)
    input_list.append(sex)
    input_list.append(age)
    input_list.append(sibsp)
    input_list.append(parch)
    input_list.append(pricerange)
    # 'res' is a list of predictions returned as the label.
    res = model.predict(np.asarray(input_list).reshape(1, -1))
    if res[0]==0:
        output = "Did not survive"
    else:
        output = "Survived"

    return output

demo = gr.Interface(
    fn=titanic,
    title="Titanic Predictive Analytics",
    description="Experiment with passenger information to predict if the passenger survived or not",
    allow_flagging="never",
    inputs=[
        gr.inputs.Number(default=1, label="ticket class (1 = 1st, 2 = 2nd, 3 = 3rd)"),
        gr.inputs.Number(default=0, label="sex (0=male, 1=female)"),
        gr.inputs.Number(default=24, label="age (years)"),
        gr.inputs.Number(default=1.0, label="# of siblings/spouses aboard"),
        gr.inputs.Number(default=1.0, label="# of children/parents aboard"),
        gr.inputs.Number(default=1.0, label="pricerange (1=cheapest, 5=most expensive)"),
        ],
    outputs="text")


demo.launch()

