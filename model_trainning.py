import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.model_selection import train_test_split, RandomizedSearchCV, cross_val_score
from sklearn.metrics import recall_score, confusion_matrix, classification_report
import pandas as pd
import numpy as np
from faker import Faker
import random
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from imblearn.over_sampling import SMOTENC
import matplotlib.pyplot as plt
import seaborn as sns
from lightgbm import LGBMClassifier
import warnings

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
df['hora_sin'] = np.sin(2 * np.pi * df['hora']/24)
df['hora_cos'] = np.cos(2 * np.pi * df['hora']/24)

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

# Nuevas características temporales
df['dia_semana'] = np.random.randint(0, 7, size=len(df))  # Simulamos día de la semana
df['mes'] = np.random.randint(1, 13, size=len(df))  # Simulamos mes
df['temporada'] = (df['mes'] % 12 + 3) // 3  # Dividimos el año en 4 temporadas

# Características de frecuencia de compra
df['frecuencia_cliente'] = df.groupby('cliente')['cliente'].transform('count')
df['frecuencia_producto'] = df.groupby('producto')['producto'].transform('count')

# Feature engineering adicional
df['cliente_mes'] = df['cliente'] * df['mes']
df['cliente_temporada'] = df['cliente'] * df['temporada']
df['frecuencia_cliente_mes'] = df.groupby(['cliente', 'mes'])['cliente'].transform('count')

# Definir características (X) y variable objetivo (y)
features = ['cliente', 'ciudad', 'supermercado', 'producto', 'estacion', 
            'hora', 'hora_sin', 'hora_cos', 'es_fin_de_semana', 
            'precio_producto_medio', 'precio_supermercado_medio',
            'precio_ciudad_medio', 'precio_estacion_medio', 'precio_hora_medio',
            'cliente_supermercado', 'cliente_producto', 'ciudad_estacion',
            'dia_semana', 'mes', 'temporada', 'frecuencia_cliente', 'frecuencia_producto',
            'cliente_mes', 'cliente_temporada', 'frecuencia_cliente_mes']
X = df[features]
y = df['precio']

# Convertir el problema a clasificación con 3 categorías
y_classes = pd.cut(y, bins=3, labels=[0, 1, 2])

# Escalar las características numéricas
scaler = StandardScaler()
X_scaled = X.copy()
numeric_features = [col for col in X.columns if X[col].dtype != 'object']
X_scaled[numeric_features] = scaler.fit_transform(X[numeric_features])

# Dividir los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_classes, test_size=0.2, random_state=42)

# Aplicar SMOTENC para balancear las clases
categorical_features_idx = [X_train.columns.get_loc(col) for col in categorical_columns if col in X_train.columns]
smote = SMOTENC(categorical_features=categorical_features_idx, random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

# Definir los modelos
xgb_model = xgb.XGBClassifier(random_state=42, eval_metric='mlogloss')
rf_model = RandomForestClassifier(random_state=42, class_weight='balanced')
lgb_model = LGBMClassifier(random_state=42, objective='multiclass', num_class=3, verbose=-1)

# Definir espacios de hiperparámetros
xgb_param_dist = {
    'n_estimators': [100, 200, 300, 400],
    'max_depth': [3, 4, 5, 6],
    'learning_rate': [0.01, 0.05, 0.1],
    'subsample': [0.6, 0.8, 1.0],
    'colsample_bytree': [0.6, 0.8, 1.0],
    'min_child_weight': [1, 3, 5],
    'reg_alpha': [0, 0.1, 0.5, 1],
    'reg_lambda': [0, 0.1, 0.5, 1]
}

rf_param_dist = {
    'n_estimators': [100, 200, 300, 400],
    'max_depth': [5, 10, 15, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2', None]
}

lgb_param_dist = {
    'n_estimators': [100, 200, 300, 400],
    'max_depth': [3, 4, 5, 6, -1],
    'learning_rate': [0.01, 0.05, 0.1, 0.2],
    'num_leaves': [31, 63, 127],
    'min_child_samples': [5, 10, 20],
    'subsample': [0.6, 0.8, 1.0],
    'colsample_bytree': [0.6, 0.8, 1.0],
    'reg_alpha': [0, 0.1, 0.5, 1],
    'reg_lambda': [0, 0.1, 0.5, 1]
}

# Realizar búsquedas aleatorias
xgb_random_search = RandomizedSearchCV(estimator=xgb_model, param_distributions=xgb_param_dist, 
                                       n_iter=20, cv=3, n_jobs=-1, verbose=1, 
                                       scoring='balanced_accuracy', random_state=42)
xgb_random_search.fit(X_train_resampled, y_train_resampled)

rf_random_search = RandomizedSearchCV(estimator=rf_model, param_distributions=rf_param_dist, 
                                      n_iter=20, cv=3, n_jobs=-1, verbose=1, 
                                      scoring='balanced_accuracy', random_state=42)
rf_random_search.fit(X_train_resampled, y_train_resampled)

lgb_random_search = RandomizedSearchCV(estimator=lgb_model, param_distributions=lgb_param_dist, 
                                       n_iter=30, cv=3, n_jobs=-1, verbose=1, 
                                       scoring='balanced_accuracy', random_state=42)
lgb_random_search.fit(X_train_resampled, y_train_resampled)

# Obtener los mejores modelos
best_xgb_model = xgb_random_search.best_estimator_
best_rf_model = rf_random_search.best_estimator_
best_lgb_model = lgb_random_search.best_estimator_

# Crear el ensemble con pesos ajustados
ensemble_model = VotingClassifier(
    estimators=[('xgb', best_xgb_model), ('rf', best_rf_model), ('lgb', best_lgb_model)],
    voting='soft',
    weights=[1, 1, 1]  # Puedes ajustar estos pesos según el rendimiento individual de cada modelo
)

# Entrenar el ensemble
ensemble_model.fit(X_train_resampled, y_train_resampled)

# Hacer predicciones
y_pred = ensemble_model.predict(X_test)

# Evaluar el modelo
recall_scores = recall_score(y_test, y_pred, average=None)
average_recall = recall_score(y_test, y_pred, average='macro')

print("\nRecall para cada clase:")
for i, score in enumerate(['bajo', 'medio', 'alto']):
    print(f"Clase {score}: {recall_scores[i]:.4f}")

print(f"\nRecall promedio: {average_recall:.4f}")

# Mostrar la matriz de confusión
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Matriz de Confusión')
plt.xlabel('Predicción')
plt.ylabel('Valor Real')
plt.show()

# Imprimir el reporte de clasificación
print("\nReporte de Clasificación:")
print(classification_report(y_test, y_pred, target_names=['bajo', 'medio', 'alto']))

# Imprimir los mejores parámetros encontrados
print("\nMejores parámetros para XGBoost:")
print(xgb_random_search.best_params_)
print("\nMejores parámetros para Random Forest:")
print(rf_random_search.best_params_)
print("\nMejores parámetros para LightGBM:")
print(lgb_random_search.best_params_)

# Calcular y mostrar la importancia de las características (usando Random Forest)
feature_importance = best_rf_model.feature_importances_
for feature, importance in zip(features, feature_importance):
    print(f"{feature}: {importance}")

# Visualizar la importancia de las características
plt.figure(figsize=(12, 8))
sorted_idx = np.argsort(feature_importance)
plt.barh(range(len(feature_importance)), feature_importance[sorted_idx])
plt.yticks(range(len(feature_importance)), [features[i] for i in sorted_idx])
plt.xlabel('Importancia')
plt.title('Importancia de las Características (Random Forest)')
plt.tight_layout()
plt.show()

# Validación cruzada
cv_scores = cross_val_score(ensemble_model, X_train_resampled, y_train_resampled, cv=5, scoring='balanced_accuracy')
print("\nPuntuaciones de validación cruzada:", cv_scores)
print(f"Media de validación cruzada: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")

# Suprimir advertencias de LightGBM
warnings.filterwarnings("ignore", category=UserWarning, module="lightgbm")

