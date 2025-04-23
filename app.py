from flask import Flask, request, jsonify
import pandas as pd
import pickle
import os
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error

app = Flask(__name__)

@app.route('/')
def index():
    return """
    <h1>Bienvenido a la API de predicción de ventas</h1>
    <p>Endpoints disponibles:</p>
    <ul>
        <li><strong>GET /api/v1/predict</strong>: Realiza una predicción pasando TV, radio y newspaper por query string.</li>
        <li><strong>GET /api/v1/retrain</strong>: Reentrena el modelo usando Advertising_new.csv y lo guarda.</li>
    </ul>
    """

@app.route('/api/v1/predict')
def predict():
    try:
        tv = float(request.args.get('TV'))
        radio = float(request.args.get('radio'))
        newspaper = float(request.args.get('newspaper'))
    except:
        return "Parámetros inválidos. Asegúrate de pasar TV, radio y newspaper como números.", 400

    test_data = pd.DataFrame([[tv, radio, newspaper]], columns=["TV", "radio", "newspaper"])

    try:
        with open("models/ad_model.pkl", "rb") as f:
            model = pickle.load(f)
    except Exception as e:
        return f"Error loading model: {str(e)}", 500

    try:
        prediction = model.predict(test_data)
        return jsonify({'prediction': float(prediction[0])})
    except Exception as e:
        return f"Prediction error: {str(e)}", 500

@app.route('/api/v1/retrain')
def retrain():
    try:
        df = pd.read_csv("Advertising_new.csv")
    except Exception as e:
        return f"Error loading dataset: {str(e)}", 500

    X = df[["TV", "radio", "newspaper"]]
    y = df["sales"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LinearRegression()
    model.fit(X_train, y_train)

    rmse = mean_squared_error(y_test, model.predict(X_test), squared=False)
    mape = mean_absolute_percentage_error(y_test, model.predict(X_test))

    try:
        os.makedirs("models", exist_ok=True)
        with open("models/ad_model.pkl", "wb") as f:
            pickle.dump(model, f)
    except Exception as e:
        return f"Error saving model: {str(e)}", 500

    return f"<h2>Modelo reentrenado correctamente</h2><p>RMSE: {rmse:.2f}, MAPE: {mape:.2%}</p>"

if __name__ == '__main__':
    app.run(debug=True)
