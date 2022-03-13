from flask import Flask, request
import numpy as np
import tensorflow as tf
import pandas as pd
import flasgger
from flasgger import Swagger
from math import *
import librosa
import statistics
import json
import joblib
from sklearn.preprocessing import MinMaxScaler
from flask import jsonify

app = Flask(__name__)
Swagger(app)

model = joblib.load(open("Final Random Forest Classifier Model.bin", 'rb'))
df_train = pd.read_csv('train.csv')



@app.route('/')
def welcome():
    return "Titanic Web App"


@app.route('/predict_survival/', methods=["POST"])
def predict_survival():
    """Let's find out whether the passenger survives or Lmao XD Ded!
    This is using docstrings for specifications.
    ---
    parameters:
      - name: sex
        in: query
        type: string
        required: true

      - name: embarked
        in: query
        type: string
        required: true

      - name: age
        in: query
        type: number
        required: true

      - name: p class
        in: query
        type: number
        required: true

      - name: siblings and spouse
        in: query
        type: number
        required: true

      - name: fare
        in: query
        type: number
        required: true

    responses:
        200:
            description: The output values
    """
    sex = request.args.get("sex")
    embarked = request.args.get("embarked")
    age = float(request.args.get("age"))
    pclass = float(request.args.get("p class"))
    sibsp = float(request.args.get("siblings and spouse"))
    fare = float(request.args.get("fare"))

    df = pd.DataFrame(
        {'Sex': ['male', 'female', 'male'], 'Embarked': ['S', 'Q', 'C'], 'Age': [1, 2, 3], 'SibSp': [1, 2, 3],
         'Pclass': [1, 2, 3], 'Fare': [1, 2, 3]})
    appended_row = {'Sex': sex, 'Embarked': embarked, 'Age': age, 'SibSp': sibsp, 'Pclass': pclass, 'Fare': fare}

    df = df.append(appended_row,ignore_index=True)

    df = pd.get_dummies(data= df, columns=['Sex', 'Embarked'])

    df = df.drop([0,1,2])
    df.drop(['Embarked_Q'], axis=1, inplace=True)

    df.reset_index(inplace=True)
    print(df.head())

    scaler = MinMaxScaler()
    scaler.fit(df_train[["Pclass", "SibSp", "Age", "Fare"]])

    df_2 = pd.concat([pd.DataFrame(scaler.transform(df[["Pclass","SibSp","Age","Fare"]]), columns=["Pclass","SibSp","Age","Fare"]), df[["Sex_female","Sex_male","Embarked_S","Embarked_C"]]], axis=1)

    print(df_2.head())

    df_2 = df_2[["Sex_female","SibSp","Pclass","Fare","Age","Embarked_S","Sex_male","Embarked_C"]]


    # return json.dumps((1))
    return json.dumps(tuple(model.predict(df_2).tolist()))


if __name__ == '__main__':
    app.run(debug=True)