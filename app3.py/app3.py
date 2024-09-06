from flask import Flask, request, render_template
import joblib
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder

app = Flask(__name__)

# Cargar el modelo entrenado y otros componentes necesarios
model = joblib.load('ensemble_model.joblib')
scaler = joblib.load('scaler.joblib')
label_encoders = joblib.load('label_encoders.joblib')
features = joblib.load('features.joblib')

@app.route('/', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        # Recoger datos del formulario
        input_data = {feature: request.form.get(feature) for feature in features}
        
        # Crear un DataFrame con los datos de entrada
        input_df = pd.DataFrame([input_data])
        
        # Aplicar Label Encoding a las características categóricas
        for col in label_encoders:
            if col in input_df.columns:
                input_df[col] = label_encoders[col].transform(input_df[col])
        
        # Aplicar el escalado a las características numéricas
        numeric_features = [col for col in input_df.columns if input_df[col].dtype != 'object']
        input_df[numeric_features] = scaler.transform(input_df[numeric_features])
        
        # Hacer la predicción
        prediction = model.predict(input_df)[0]
        
        # Obtener el supermercado correspondiente
        supermercado = input_data['supermercado']
        
        return render_template('index.html', prediction=prediction, supermercado=supermercado)
    
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)