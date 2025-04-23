import pandas as pd
from sklearn.linear_model import LinearRegression
import joblib

# Cargar el dataset CSV
df = pd.read_csv("Advertising_new.csv")

# Variables predictoras (X) y variable objetivo (y)
X = df[["TV", "radio", "newspaper"]]
y = df["sales"]

# Entrenar modelo de regresión lineal
model = LinearRegression()
model.fit(X, y)

# Guardar el modelo entrenado en formato joblib
joblib.dump(model, "models/ad_model.pkl")

print("✅ Modelo entrenado y guardado correctamente en 'models/ad_model.pkl'")
