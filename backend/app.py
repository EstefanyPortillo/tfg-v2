from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
import cv2
import numpy as np
import os

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'temp'  # Carpeta temporal para las imágenes subidas

# Cargar el modelo entrenado
modelo = load_model("modelo_optimizado_v3.keras")

# Función para preprocesar la imagen antes de la predicción
def preprocesar_imagen(imagen_ruta, tamano_imagen=224):
    imagen = cv2.imread(imagen_ruta)
    imagen = cv2.resize(imagen, (tamano_imagen, tamano_imagen))  # Redimensionar
    imagen = imagen / 255.0  # Normalizar
    imagen = np.expand_dims(imagen, axis=0)  # Añadir dimensión para el modelo
    return imagen

# Ruta principal para recibir la imagen y hacer la predicción
@app.route('/upload', methods=['POST'])
def upload():
    # Verificar que se haya enviado un archivo
    if 'file' not in request.files:
        return jsonify({"error": "No se encontró ningún archivo"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No se seleccionó ningún archivo"}), 400

    # Guardar la imagen temporalmente
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(file_path)

    # Preprocesar la imagen y hacer la predicción
    imagen_preprocesada = preprocesar_imagen(file_path)
    prediccion = modelo.predict(imagen_preprocesada)
    clase = np.argmax(prediccion, axis=1)[0]
    resultado = "Lesión benigna" if clase == 0 else "Lesión maligna"
    
    # Eliminar la imagen después de la predicción
    os.remove(file_path)

    # Devolver el resultado en formato JSON
    return jsonify({"resultado": resultado})

if __name__ == "__main__":
    # Asegurarse de que la carpeta de almacenamiento temporal existe
    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])
    app.run(debug=True)
