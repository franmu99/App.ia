import xgboost as xgb
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error
import pandas as pd
import numpy as np
from faker import Faker
import random
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Initialize Faker
fake = Faker()

# Define the number of rows for our dataset
n_rows = 1000  # 1 million rows

# Create lists for our categorical variables
clientes = [fake.name() for _ in range(1000)]  # 10,000 unique customers
ciudades = ['Madrid', 'Barcelona', 'Valencia', 'Sevilla', 'Zaragoza', 'Málaga', 'Murcia', 'Palma', 'Las Palmas', 'Bilbao']
supermercados = ['Mercadona', 'Carrefour', 'Dia', 'Lidl', 'Aldi', 'Eroski', 'Alcampo', 'El Corte Inglés', 'Consum', 'Bon Preu']
productos = ['Leche', 'Pan', 'Huevos', 'Arroz', 'Pasta', 'Tomates', 'Manzanas', 'Pollo', 'Atún', 'Yogur']
estaciones = ['Primavera', 'Verano', 'Otoño', 'Invierno']

# Generate the dataset
data = {
    'cliente': [random.choice(clientes) for _ in range(n_rows)],
    'ciudad': [random.choice(ciudades) for _ in range(n_rows)],
    'supermercado': [random.choice(supermercados) for _ in range(n_rows)],
    'producto': [random.choice(productos) for _ in range(n_rows)],
    'precio': [round(random.uniform(0.5, 50.0), 2) for _ in range(n_rows)],
    'estacion': [random.choice(estaciones) for _ in range(n_rows)],
    'hora': [f"{random.randint(8, 21):02d}:00" for _ in range(n_rows)]
}

# Create DataFrame
df = pd.DataFrame(data)

# Save to CSV (optional)
df.to_csv('dataset_supermercados.csv', index=False)

print(f"Dataset created with {n_rows} rows and {len(df.columns)} columns.")
print(df.head())

# Análisis de correlación con la variable precio
# Convertir las variables categóricas en numéricas usando label encoding
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
df_encoded = df.copy()
categorical_columns = ['cliente', 'ciudad', 'supermercado', 'producto', 'estacion']
for col in categorical_columns:
    df_encoded[col] = le.fit_transform(df_encoded[col])

# Convertir la columna 'hora' a formato numérico (horas desde medianoche)
df_encoded['hora'] = pd.to_datetime(df_encoded['hora'], format='%H:%M').dt.hour

# Calcular la matriz de correlaciones
correlation_matrix = df_encoded[['cliente', 'ciudad', 'supermercado', 'producto', 'estacion', 'hora', 'precio']].corr()

# Extraer las correlaciones con la variable 'precio'
price_correlations = correlation_matrix['precio'].drop('precio')

print("\nCorrelaciones de cada variable con 'precio':")
print(price_correlations)

# Visualizar la matriz de correlaciones
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1, center=0)
plt.title('Matriz de Correlaciones')
plt.show()

# Calcular la media de precios por supermercado
precio_medio_supermercado = df.groupby('supermercado')['precio'].mean().sort_values(ascending=False)

# Crear un gráfico de barras para visualizar la media de precios por supermercado
plt.figure(figsize=(12, 6))
precio_medio_supermercado.plot(kind='bar')
plt.title('Precio Medio por Supermercado')
plt.xlabel('Supermercado')
plt.ylabel('Precio Medio')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()

# Añadir etiquetas de valor encima de cada barra
for i, v in enumerate(precio_medio_supermercado):
    plt.text(i, v, f'{v:.2f}', ha='center', va='bottom')

plt.show()

print("\nPrecio medio por supermercado:")
print(precio_medio_supermercado)

# Preprocesamiento
le = LabelEncoder()
categorical_columns = ['cliente', 'ciudad', 'supermercado', 'producto', 'estacion']
for col in categorical_columns:
    df[col] = le.fit_transform(df[col])

# Convertir la columna 'hora' a formato numérico y crear características cíclicas
df['hora'] = pd.to_datetime(df['hora'], format='%H:%M').dt.hour
df['hora_sin'] = np.sin(2 * np.pi * df['hora']/24)
df['hora_cos'] = np.cos(2 * np.pi * df['hora']/24)

# Feature engineering
df['es_fin_de_semana'] = (df['hora'] >= 10) & (df['hora'] <= 20)
df['precio_producto_medio'] = df.groupby('producto')['precio'].transform('mean')
df['precio_supermercado_medio'] = df.groupby('supermercado')['precio'].transform('mean')

# Definir características (X) y variable objetivo (y)
features = ['cliente', 'ciudad', 'supermercado', 'producto', 'estacion', 
            'hora', 'hora_sin', 'hora_cos', 'es_fin_de_semana', 
            'precio_producto_medio', 'precio_supermercado_medio']
X = df[features]
y = df['precio']

# Escalar las características numéricas
scaler = StandardScaler()
X_scaled = X.copy()
X_scaled[['hora', 'hora_sin', 'hora_cos', 'precio_producto_medio', 'precio_supermercado_medio']] = scaler.fit_transform(X[['hora', 'hora_sin', 'hora_cos', 'precio_producto_medio', 'precio_supermercado_medio']])

# Dividir los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Definir el modelo base
model = xgb.XGBRegressor(random_state=42)

# Definir los parámetros para la búsqueda en cuadrícula
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [3, 4, 5],
    'learning_rate': [0.01, 0.1, 0.3],
    'subsample': [0.8, 0.9, 1.0],
    'colsample_bytree': [0.8, 0.9, 1.0]
}

# Realizar la búsqueda en cuadrícula
grid_search = GridSearchCV(estimator=model, param_grid=param_grid, 
                           cv=3, n_jobs=-1, verbose=2, scoring='neg_mean_squared_error')
grid_search.fit(X_train, y_train)

# Obtener el mejor modelo
best_model = grid_search.best_estimator_

# Hacer predicciones
y_pred = best_model.predict(X_test)

# Evaluar el modelo
mse = mean_squared_error(y_test, y_pred)
rmse = mse ** 0.5

print(f"Mejor RMSE: {rmse}")
print(f"Mejores parámetros: {grid_search.best_params_}")

# Importancia de las características
feature_importance = best_model.feature_importances_
for feature, importance in zip(features, feature_importance):
    print(f"{feature}: {importance}")

# Visualizar la importancia de las características
plt.figure(figsize=(10, 6))
sorted_idx = np.argsort(feature_importance)
plt.barh(range(len(feature_importance)), feature_importance[sorted_idx])
plt.yticks(range(len(feature_importance)), [features[i] for i in sorted_idx])
plt.xlabel('Importancia')
plt.title('Importancia de las Características')
plt.tight_layout()
plt.show()

