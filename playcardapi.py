import os
import pandas as pd
import pickle
from flask import Flask, request


# load model
model = pickle.load( open('playcardsapi.pkl', 'rb'))
# instanciate flask
app = Flask( __name__ )


@app.route('/')
def index():
    return "<h1>Excutando Pyton</h1>"

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
    df_raw['prediction'] = pred

    return df_raw.to_json(orient='records')

if __name__ == '__main__':
    # start flask
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)