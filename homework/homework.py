#
# En este dataset se desea pronosticar el precio de vhiculos usados. El dataset
# original contiene las siguientes columnas:
#
# - Car_Name: Nombre del vehiculo.
# - Year: Año de fabricación.
# - Selling_Price: Precio de venta.
# - Present_Price: Precio actual.
# - Driven_Kms: Kilometraje recorrido.
# - Fuel_type: Tipo de combustible.
# - Selling_Type: Tipo de vendedor.
# - Transmission: Tipo de transmisión.
# - Owner: Número de propietarios.
#
# El dataset ya se encuentra dividido en conjuntos de entrenamiento y prueba
# en la carpeta "files/input/".
#
# Los pasos que debe seguir para la construcción de un modelo de
# pronostico están descritos a continuación.
#
#
# Paso 1.
# Preprocese los datos.
# - Cree la columna 'Age' a partir de la columna 'Year'.
#   Asuma que el año actual es 2021.
# - Elimine las columnas 'Year' y 'Car_Name'.
#
#
# Paso 2.
# Divida los datasets en x_train, y_train, x_test, y_test.
#
#
# Paso 3.
# Cree un pipeline para el modelo de clasificación. Este pipeline debe
# contener las siguientes capas:
# - Transforma las variables categoricas usando el método
#   one-hot-encoding.
# - Escala las variables numéricas al intervalo [0, 1].
# - Selecciona las K mejores entradas.
# - Ajusta un modelo de regresion lineal.
#
#
# Paso 4.
# Optimice los hiperparametros del pipeline usando validación cruzada.
# Use 10 splits para la validación cruzada. Use el error medio absoluto
# para medir el desempeño modelo.
#
#
# Paso 5.
# Guarde el modelo (comprimido con gzip) como "files/models/model.pkl.gz".
# Recuerde que es posible guardar el modelo comprimido usanzo la libreria gzip.
#
#
# Paso 6.
# Calcule las metricas r2, error cuadratico medio, y error absoluto medio
# para los conjuntos de entrenamiento y prueba. Guardelas en el archivo
# files/output/metrics.json. Cada fila del archivo es un diccionario con
# las metricas de un modelo. Este diccionario tiene un campo para indicar
# si es el conjunto de entrenamiento o prueba. Por ejemplo:
#
# {'type': 'metrics', 'dataset': 'train', 'r2': 0.8, 'mse': 0.7, 'mad': 0.9}
# {'type': 'metrics', 'dataset': 'test', 'r2': 0.7, 'mse': 0.6, 'mad': 0.8}
#

import os
import json
import gzip
import pickle
from pathlib import Path

import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score, mean_squared_error, median_absolute_error

# Paso 1 — Lectura y preprocesamiento

train_pd = pd.read_csv("files/input/train_data.csv.zip", compression="zip").copy()
test_pd = pd.read_csv("files/input/test_data.csv.zip", compression="zip").copy()

# Crear Age
train_pd["Age"] = 2021 - train_pd["Year"]
test_pd["Age"] = 2021 - test_pd["Year"]

# Eliminar columnas
train_pd = train_pd.drop(columns=["Year", "Car_Name"])
test_pd = test_pd.drop(columns=["Year", "Car_Name"])

# Paso 2 — Separar X e y

X_train = train_pd.drop(columns=["Present_Price"])
y_train = train_pd["Present_Price"]

X_test = test_pd.drop(columns=["Present_Price"])
y_test = test_pd["Present_Price"]

# Paso 3 — Pipeline (OHE + MinMax + KBest + LinearRegression)

cat_cols = ["Fuel_Type", "Selling_type", "Transmission"]
num_cols = [c for c in X_train.columns if c not in cat_cols]

preprocesador = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(), cat_cols),
        ("num", MinMaxScaler(), num_cols),
    ]
)

pipe = Pipeline(
    steps=[
        ("pre", preprocesador),
        ("selector", SelectKBest(score_func=f_regression)),
        ("reg", LinearRegression()),
    ]
)

# Paso 4 — GridSearchCV para ajustar hiperparámetros

param_grid = {
    "selector__k": range(1, 15),
    "reg__fit_intercept": [True, False],
    "reg__positive": [True, False],
}

grid = GridSearchCV(
    estimator=pipe,
    param_grid=param_grid,
    cv=10,
    scoring="neg_mean_absolute_error",
    n_jobs=-1,
    refit=True,
)

grid.fit(X_train, y_train)

# Paso 5 — Guardar modelo .pkl.gz

os.makedirs("files/models", exist_ok=True)

with gzip.open("files/models/model.pkl.gz", "wb") as f:
    pickle.dump(grid, f)

# Paso 6 — Métricas (train y test)

pred_train = grid.predict(X_train)
pred_test = grid.predict(X_test)

train_metrics = {
    "type": "metrics",
    "dataset": "train",
    "r2": float(r2_score(y_train, pred_train)),
    "mse": float(mean_squared_error(y_train, pred_train)),
    "mad": float(median_absolute_error(y_train, pred_train)),
}

test_metrics = {
    "type": "metrics",
    "dataset": "test",
    "r2": float(r2_score(y_test, pred_test)),
    "mse": float(mean_squared_error(y_test, pred_test)),
    "mad": float(median_absolute_error(y_test, pred_test)),
}

# Guardar metrics.json

Path("files/output").mkdir(parents=True, exist_ok=True)

with open("files/output/metrics.json", "w", encoding="utf-8") as f:
    f.write(json.dumps(train_metrics) + "\n")
    f.write(json.dumps(test_metrics) + "\n")
