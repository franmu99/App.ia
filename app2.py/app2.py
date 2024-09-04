from flask import Flask, render_template, request
import joblib
import pandas as pd
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)

# Cargar el modelo entrenado
modelo = joblib.load('modelo_random_forest.joblib')

@app.route('/', methods=['GET', 'POST'])
def index():
    prediccion_precio = None
    prediccion_supermercado = None
    if request.method == 'POST':
        # Obtener datos del formulario
        producto = request.form['producto']
        edad = float(request.form['edad'])
        ingreso = float(request.form['ingreso'])
        dia_semana = request.form['dia_semana']
        temporada = request.form['temporada']

        # Crear un DataFrame con los datos de entrada
        datos = pd.DataFrame({
            'producto': [producto],
            'edad_cliente': [edad],
            'ingreso_cliente': [ingreso],
            'dia_semana': [dia_semana],
            'temporada': [temporada]
        })

        # Realizar la predicción
        predicciones = modelo.predict(datos)
        prediccion_precio = predicciones[0][0]  # Asumiendo que el precio es la primera salida
        prediccion_supermercado = predicciones[0][1]  # Asumiendo que el supermercado es la segunda salida

    return render_template('index.html', prediccion_precio=prediccion_precio, prediccion_supermercado=prediccion_supermercado)

if __name__ == '__main__':
    app.run(debug=True)