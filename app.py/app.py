import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import graphviz
from sklearn.tree import export_graphviz
import random
from datetime import datetime, timedelta
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt

def generar_fechas(n):
    fecha_inicio = datetime(2023, 1, 1)
    fechas = [fecha_inicio + timedelta(days=random.randint(0, 364)) for _ in range(n)]
    return fechas

def obtener_temporada(fecha):
    mes = fecha.month
    if mes in [12, 1, 2]:
        return 'Invierno'
    elif mes in [3, 4, 5]:
        return 'Primavera'
    elif mes in [6, 7, 8]:
        return 'Verano'
    else:
        return 'Otoño'

def cargar_datos():
    # Aquí deberías cargar tus datos reales
    n = 60
    fechas = generar_fechas(n)
    data = pd.DataFrame({
        'producto': ['manzana', 'leche', 'pan', 'manzana', 'leche', 'pan'] * 10,
        'precio': np.random.uniform(0.5, 3.0, n),
        'supermercado': np.random.choice(['A', 'B', 'C'], n),
        'edad_cliente': np.random.randint(18, 70, n),
        'ingreso_cliente': np.random.randint(20000, 100000, n),
        'fecha': fechas,
        'dia_semana': [fecha.strftime('%A') for fecha in fechas],
        'temporada': [obtener_temporada(fecha) for fecha in fechas]
    })
    return data

def preprocesar_datos(data):
    X = data[['producto', 'edad_cliente', 'ingreso_cliente', 'supermercado', 'dia_semana', 'temporada']]
    y = data['precio']
    
    return train_test_split(X, y, test_size=0.2, random_state=42)



def crear_modelo():
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), ['edad_cliente', 'ingreso_cliente']),
            ('cat_producto', OneHotEncoder(drop='first', handle_unknown='ignore'), ['producto']),
            ('cat_supermercado', OneHotEncoder(drop='first', handle_unknown='ignore'), ['supermercado']),
            ('cat_dia_semana', OneHotEncoder(drop='first', handle_unknown='ignore'), ['dia_semana']),
            ('cat_temporada', OneHotEncoder(drop='first', handle_unknown='ignore'), ['temporada'])
        ])

    rf = RandomForestRegressor(random_state=42)
    
    model = Pipeline([
        ('preprocessor', preprocessor),
        ('regressor', rf)
    ])
    
    param_grid = {
        'regressor__n_estimators': [50, 100, 200],
        'regressor__max_depth': [None, 10, 20, 30],
        'regressor__min_samples_split': [2, 5, 10],
        'regressor__min_samples_leaf': [1, 2, 4]
    }
    
    grid_search = GridSearchCV(model, param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
    
    return grid_search

def entrenar_modelo(model, X_train, y_train):
    model.fit(X_train, y_train)
    print("Mejores parámetros encontrados:")
    print(model.best_params_)
    return model.best_estimator_


    
def evaluar_modelo(model, X_test, y_test):
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    print(f"Error cuadrático medio (MSE): {mse:.2f}")
    print(f"Raíz del error cuadrático medio (RMSE): {rmse:.2f}")
def analizar_residuos(model, X_test, y_test):
    # Realizar predicciones
    y_pred = model.predict(X_test)
    
    # Calcular residuos
    residuos = y_test - y_pred
    
    # Crear gráfico de dispersión de residuos
    plt.figure(figsize=(10, 6))
    plt.scatter(y_pred, residuos, alpha=0.5)
    plt.xlabel('Valores predichos')
    plt.ylabel('Residuos')
    plt.title('Análisis de Residuos')
    plt.axhline(y=0, color='r', linestyle='--')
    
    # Añadir una línea de tendencia
    z = np.polyfit(y_pred, residuos, 1)
    p = np.poly1d(z)
    plt.plot(y_pred, p(y_pred), "r--", alpha=0.8)
    
    plt.tight_layout()
    plt.savefig('analisis_residuos.png')
    plt.close()
    
    print("El gráfico de análisis de residuos ha sido guardado.")
    
    # Calcular y mostrar estadísticas de los residuos
    print(f"Media de los residuos: {np.mean(residuos):.2f}")
    print(f"Desviación estándar de los residuos: {np.std(residuos):.2f}")

    

def visualizar_arbol(model, feature_names):
    # Obtener uno de los árboles del bosque (por ejemplo, el primero)
    tree = model.named_steps['regressor'].estimators_[0]
    
    # Exportar el árbol a formato DOT
    dot_data = export_graphviz(tree, 
                               feature_names=feature_names,
                               filled=True, 
                               rounded=True,
                               special_characters=True)
    
    # Crear el gráfico
    graph = graphviz.Source(dot_data)
    
    # Guardar y mostrar el gráfico
    graph.render("arbol_decision", format="png", cleanup=True)
    print("El árbol de decisión ha sido guardado como 'arbol_decision.png'")

def predecir_mejor_opcion(model, producto, edad, ingreso, supermercados):
    resultados = []
    for supermercado in supermercados:
        precio_predicho = model.predict(pd.DataFrame({
            'producto': [producto],
            'edad_cliente': [edad],
            'ingreso_cliente': [ingreso],
            'supermercado': [supermercado],
            'dia_semana': ['Lun'],  
            'temporada': ['Primavera']
        }))[0]
        resultados.append((supermercado, precio_predicho))
    
    mejor_opcion = min(resultados, key=lambda x: x[1])
    return mejor_opcion

def main():
    data = cargar_datos()
    X_train, X_test, y_train, y_test = preprocesar_datos(data)
    
    modelo = crear_modelo()
    modelo_entrenado = entrenar_modelo(modelo, X_train, y_train)
    evaluar_modelo(modelo_entrenado, X_test, y_test)
    
    # Visualizar un árbol del Random Forest
    feature_names = modelo_entrenado.named_steps['preprocessor'].get_feature_names_out()
    visualizar_arbol(modelo_entrenado, feature_names)
    
    # Ejemplo de predicción para un cliente y producto específico
    producto = 'manzana'
    edad_cliente = 40
    ingreso_cliente = 45000
    supermercados = ['A', 'B', 'C']
    
    mejor_supermercado, mejor_precio = predecir_mejor_opcion(
        modelo_entrenado, producto, edad_cliente, ingreso_cliente, supermercados
    )
    
    print(f"Para un cliente de {edad_cliente} años y un ingreso de ${ingreso_cliente}, "
          f"buscando {producto}:")
    print(f"El supermercado más barato es {mejor_supermercado} con un precio estimado de ${mejor_precio:.2f}")

if __name__ == "__main__":
    main()