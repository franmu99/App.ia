import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder

# Cargar los datos
df = pd.read_csv('dataset_supermercados.csv')

# Preprocesamiento
le = LabelEncoder()
categorical_columns = ['cliente', 'ciudad', 'supermercado', 'producto', 'estacion']
for col in categorical_columns:
    df[col] = le.fit_transform(df[col])

# Convertir la columna 'hora' a formato numérico (horas desde medianoche)
df['hora'] = pd.to_datetime(df['hora']).dt.hour

# Definir características (X) y variable objetivo (y)
X = df[['cliente', 'ciudad', 'supermercado', 'producto', 'estacion', 'hora']]
y = df['precio']

# Dividir los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Crear y entrenar el modelo XGBoost
model = xgb.XGBRegressor(random_state=42)
model.fit(X_train, y_train)

# Hacer predicciones
y_pred = model.predict(X_test)

# Evaluar el modelo
mse = mean_squared_error(y_test, y_pred)
rmse = mse ** 0.5

print(f"RMSE: {rmse}")

# Importancia de las características
feature_importance = model.feature_importances_
for feature, importance in zip(X.columns, feature_importance):
    print(f"{feature}: {importance}")