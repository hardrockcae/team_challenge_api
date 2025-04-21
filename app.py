from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
import os
import pickle
from sklearn.linear_model import Lasso
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error

app = Flask(__name__)

# Landing page
@app.route("/", methods=["GET"])
def home():
    return (
        "<h1>Advertising Model API</h1>"
        "<p>Endpoints disponibles:</p>"
        "<ul>"
        "<li>/api/v1/predict?TV=valor&radio=valor&newspaper=valor</li>"
        "<li>/api/v1/retrain</li>"
        "<li>/api/v1/extra (comentado)</li>"
        "</ul>"
    )

def get_arg(*names, type=float):
    for name in names:
        val = request.args.get(name, type=type)
        if val is not None:
            return val
    return None

@app.route('/api/v1/predict', methods=['GET'])
def predict():
    tv = get_arg('TV', 'tv')
    radio = get_arg('radio', 'Radio')
    newspaper = get_arg('newspaper', 'Newspaper')

    if tv is None or radio is None or newspaper is None:
        return "Args empty, not enough data to predict", 400

    test_data = pd.DataFrame([[tv, radio, newspaper]], columns=["TV", "radio", "newspaper"])

    try:
        with open("ad_model.pkl", "rb") as f:
            model = pickle.load(f)
    except Exception as e:
        return f"Error loading model: {str(e)}", 500

    try:
        prediction = model.predict(test_data)
        return jsonify({'prediction': float(prediction[0])})
    except Exception as e:
        return f"Prediction error: {str(e)}", 500

@app.route('/api/v1/retrain', methods=['GET'])
def retrain():
    if os.path.exists("data/Advertising_new.csv"):
        data = pd.read_csv("data/Advertising_new.csv")
        X_train, X_test, y_train, y_test = train_test_split(
            data.drop(columns=['sales']),
            data['sales'],
            test_size=0.2,
            random_state=42
        )

        model = Lasso(alpha=6000)
        model.fit(X_train, y_train)

        rmse = np.sqrt(mean_squared_error(y_test, model.predict(X_test)))
        mape = mean_absolute_percentage_error(y_test, model.predict(X_test))

        with open("ad_model.pkl", "wb") as f:
            pickle.dump(model, f)

        return f"Model retrained. RMSE: {rmse:.2f}, MAPE: {mape:.2%}"
    else:
        return "<h2>New data for retrain NOT FOUND. Nothing done!</h2>"

# Endpoint extra (comentado para redespliegue)
# @app.route('/api/v1/extra', methods=['GET'])
# def extra():
#     return "Este es un endpoint extra para la demo del redespliegue."
