# Advertising Model API

API Flask que permite hacer predicciones y reentrenar un modelo de regresión Lasso con datos de publicidad.

## Endpoints

- `/api/v1/predict?TV=valor&radio=valor&newspaper=valor` → Devuelve una predicción
- `/api/v1/retrain` → Reentrena el modelo con datos nuevos
- `/api/v1/extra` → Comentado, útil para mostrar redespliegue en demo
