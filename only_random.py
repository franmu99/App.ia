import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.model_selection import train_test_split, RandomizedSearchCV, cross_val_score, GridSearchCV
from sklearn.metrics import recall_score, confusion_matrix, classification_report
import pandas as pd
import numpy as np
from faker import Faker
import random
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from imblearn.combine import SMOTEENN
import matplotlib.pyplot as plt
import seaborn as sns
from lightgbm import LGBMClassifier
import warnings
from imblearn.combine import SMOTETomek
from imblearn.over_sampling import SMOTENC
from imblearn.under_sampling import RandomUnderSampler
from collections import Counter
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from scipy.stats import randint, uniform
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier
from scipy.stats import uniform, randint
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.calibration import CalibratedClassifierCV
from imblearn.over_sampling import SMOTE

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

# Preprocesamiento y feature engineering
le = LabelEncoder()
categorical_columns = ['cliente', 'ciudad', 'supermercado', 'producto', 'estacion']
for col in categorical_columns:
    df[col] = le.fit_transform(df[col])

df['hora'] = pd.to_datetime(df['hora'], format='%H:%M').dt.hour


# Feature engineering adicional
df['es_fin_de_semana'] = (df['hora'] >= 10) & (df['hora'] <= 20)
df['precio_producto_medio'] = df.groupby('producto')['precio'].transform('mean')
df['precio_supermercado_medio'] = df.groupby('supermercado')['precio'].transform('mean')
df['precio_ciudad_medio'] = df.groupby('ciudad')['precio'].transform('mean')
df['precio_estacion_medio'] = df.groupby('estacion')['precio'].transform('mean')
df['precio_hora_medio'] = df.groupby('hora')['precio'].transform('mean')

# Características de interacción
df['cliente_supermercado'] = df['cliente'] * df['supermercado']
df['cliente_producto'] = df['cliente'] * df['producto']
df['ciudad_estacion'] = df['ciudad'] * df['estacion']




# Definir características (X) y variable objetivo (y)
features = ['cliente', 'ciudad', 'supermercado', 'producto', 'estacion', 
            'hora', 'es_fin_de_semana', 'precio_producto_medio', 
            'precio_supermercado_medio', 'precio_ciudad_medio', 
            'precio_estacion_medio', 'precio_hora_medio',
            'cliente_supermercado', 'cliente_producto', 'ciudad_estacion']

X = df[features]
y = pd.cut(df['precio'], bins=3, labels=[0, 1, 2])

# Escalar las características numéricas
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Dividir los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_encoded, test_size=0.2, random_state=42)

# Usar SMOTE en lugar de SMOTEENN
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

print("Distribución de clases original en el conjunto de entrenamiento:")
print(pd.Series(y_train).value_counts(normalize=True))

print("\nDistribución de clases después de aplicar SMOTE:")
print(pd.Series(y_resampled).value_counts(normalize=True))

# XGBoost con parámetros
xgb_param_dist = {
    'n_estimators': randint(500, 1500),
    'max_depth': randint(4, 8),
    'learning_rate': uniform(0.01, 0.2),
    'subsample': uniform(0.7, 0.3),
    'colsample_bytree': uniform(0.7, 0.3),
    'min_child_weight': randint(1, 5),
    'gamma': uniform(0, 0.3)
}

xgb_random = RandomizedSearchCV(
    XGBClassifier(random_state=42, eval_metric='mlogloss', objective='multi:softprob'),
    param_distributions=xgb_param_dist,
    n_iter=100,
    cv=5,
    n_jobs=-1,
    verbose=1,
    random_state=42
)

# Ajustar el modelo con los datos balanceados
xgb_random.fit(X_resampled, y_resampled)

# Obtener el mejor modelo
best_xgb_model = xgb_random.best_estimator_

# Calibración del modelo
calibrated_xgb = CalibratedClassifierCV(best_xgb_model, cv=5, method='isotonic')
calibrated_xgb.fit(X_resampled, y_resampled)

# Ensemble con Random Forest
rf_model = RandomForestClassifier(n_estimators=500, random_state=42)
ensemble = VotingClassifier(
    estimators=[('xgb', calibrated_xgb), ('rf', rf_model)],
    voting='soft'
)
ensemble.fit(X_resampled, y_resampled)

# Hacer predicciones en el conjunto de prueba original
y_pred = ensemble.predict(X_test)

# Evaluar el modelo
print("\nMejores parámetros encontrados:")
print(xgb_random.best_params_)

print("\nRecall para cada clase:")
recall_scores = recall_score(y_test, y_pred, average=None)
for i, score in enumerate(le.classes_):
    print(f"Clase {score}: {recall_scores[i]:.4f}")

print(f"\nRecall promedio: {recall_score(y_test, y_pred, average='macro'):.4f}")

print("\nReporte de Clasificación:")
print(classification_report(y_test, y_pred, target_names=le.classes_))

# Mostrar la matriz de confusión
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Matriz de Confusión')
plt.xlabel('Predicción')
plt.ylabel('Valor Real')
plt.show()

# Importancia de las características
feature_importance = best_xgb_model.feature_importances_
feature_importance_df = pd.DataFrame({'feature': X.columns, 'importance': feature_importance})
feature_importance_df = feature_importance_df.sort_values('importance', ascending=False)

plt.figure(figsize=(12, 8))
sns.barplot(x='importance', y='feature', data=feature_importance_df)
plt.title('Importancia de las Características')
plt.xlabel('Importancia')
plt.ylabel('Característica')
plt.tight_layout()
plt.show()

# Validación cruzada
cv_scores = cross_val_score(best_xgb_model, X_scaled, y, cv=5, scoring='balanced_accuracy')
print("\nPuntuaciones de validación cruzada:", cv_scores)
print(f"Media de validación cruzada: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")


