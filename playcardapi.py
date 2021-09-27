import os
import pandas as pd
import numpy as np
import pickle
from flask import Flask, request
from flask_cors import CORS, cross_origin
from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


# load model
model = pickle.load( open('playcardsapi.pkl', 'rb'))
# instanciate flask
app = Flask( __name__ )
cors = CORS(app, resources={"/*": {"origins": "*"}})

@app.route('/')
def index():
    return "<h1>Machine Learning - Árvore de decisão - v1.0a2</h1>"

@app.route('/predict', methods=['POST'])
def predict():
    test_json = request.get_json()
    # collect data
    if test_json:
        if isinstance(test_json, dict): # unique value
            df_raw = pd.DataFrame(test_json, index=[0])
        else:
            df_raw = pd.DataFrame(test_json,columns=test_json[0].keys())

    # prediction
    pred = model.predict(df_raw)
    df_raw['resultado'] = pred
    # df_raw.headers.add("Access-Control-Allow-Origin", "*")
    return df_raw.to_json(orient='records')

if __name__ == '__main__':
    # start flask
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)