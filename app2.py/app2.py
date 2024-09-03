from flask import Flask, render_template, request
import joblib
import pandas as pd
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)

# Cargar el modelo entrenado
modelo = joblib.load('modelo_random_forest.joblib')

@app.route('/', methods=['GET', 'POST'])
def index():
    prediccion = None
    if request.method == 'POST':
        # Obtener datos del formulario
        producto = request.form['producto']
        edad = float(request.form['edad'])
        ingreso = float(request.form['ingreso'])
        supermercado = request.form['supermercado']
        dia_semana = request.form['dia_semana']
        temporada = request.form['temporada']

        # Crear un DataFrame con los datos de entrada
        datos = pd.DataFrame({
            'producto': [producto],
            'edad_cliente': [edad],
            'ingreso_cliente': [ingreso],
            'supermercado': [supermercado],
            'dia_semana': [dia_semana],
            'temporada': [temporada]
        })

        # Realizar la predicción
        prediccion = modelo.predict(datos)[0]

    return render_template('index.html', prediccion=prediccion)

if __name__ == '__main__':
    app.run(debug=True)