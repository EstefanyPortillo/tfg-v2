from flask import Flask, request, jsonify, send_file
from tensorflow.keras.models import load_model, Model
import cv2
import numpy as np
import tensorflow as tf
from skimage.metrics import structural_similarity as ssim
import os
from flask_cors import CORS
from PIL import Image
import io

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})  # Permitir todas las solicitudes de origen

modelo = load_model("modelo_optimizado_v3.keras")

# Preprocesar la imagen
def preprocesar_imagen(imagen, tamano_imagen=224):
    imagen = cv2.resize(imagen, (tamano_imagen, tamano_imagen))
    imagen = imagen / 255.0
    imagen = np.expand_dims(imagen, axis=0)
    return imagen

# Calcular porcentaje de similitud
def calcular_porcentaje_similitud(imagen1, imagen2):
    imagen1_gray = cv2.cvtColor(imagen1, cv2.COLOR_BGR2GRAY)
    imagen2_gray = cv2.cvtColor(imagen2, cv2.COLOR_BGR2GRAY)
    score, _ = ssim(imagen1_gray, imagen2_gray, full=True)
    return score * 100

# Hacer predicción con el modelo
def predecir_imagen(modelo, imagen):
    imagen_preprocesada = preprocesar_imagen(imagen)
    prediccion = modelo.predict(imagen_preprocesada)
    clase = "Lesión benigna" if np.argmax(prediccion, axis=1)[0] == 0 else "Lesión maligna"
    return clase

# Generar Grad-CAM
def grad_cam(model, img_array, layer_name="conv5_block3_out"):
    grad_model = Model(
        inputs=[model.inputs],
        outputs=[model.get_layer(layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        loss = predictions[:, np.argmax(predictions[0])]

    output = conv_outputs[0]
    grads = tape.gradient(loss, conv_outputs)[0]
    weights = tf.reduce_mean(grads, axis=(0, 1))

    cam = np.zeros(output.shape[:2], dtype=np.float32)
    for i, w in enumerate(weights):
        cam += w * output[:, :, i]

    cam = np.maximum(cam, 0)
    cam = cam / cam.max()
    return cam

# Ruta para comparar dos imágenes
@app.route('/comparar', methods=['POST'])
def comparar():
    file1 = request.files.get('imagen1')
    file2 = request.files.get('imagen2')

    if not file1 or not file2:
        return jsonify({"error": "Ambas imágenes son requeridas"}), 400

    imagen1 = cv2.imdecode(np.frombuffer(file1.read(), np.uint8), cv2.IMREAD_COLOR)
    imagen2 = cv2.imdecode(np.frombuffer(file2.read(), np.uint8), cv2.IMREAD_COLOR)

    porcentaje_similitud = calcular_porcentaje_similitud(imagen1, imagen2)
    prediccion1 = predecir_imagen(modelo, imagen1)
    prediccion2 = predecir_imagen(modelo, imagen2)

    return jsonify({
        "similitud": f"{porcentaje_similitud:.2f}%",
        "prediccion1": prediccion1,
        "prediccion2": prediccion2
    })

# Ruta para obtener el Grad-CAM de una imagen
@app.route('/gradcam', methods=['POST'])
def obtener_gradcam():
    file = request.files.get('imagen')

    if not file:
        return jsonify({"error": "Se requiere una imagen"}), 400

    img = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)
    img = cv2.resize(img, (224, 224))
    img_array = np.expand_dims(img / 255.0, axis=0)

    heatmap = grad_cam(modelo, img_array)

    img = cv2.resize(img, (heatmap.shape[1], heatmap.shape[0]))
    heatmap = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
    superimposed_img = cv2.addWeighted(img, 0.6, heatmap, 0.4, 0)

    superimposed_img = cv2.cvtColor(superimposed_img, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(superimposed_img)
    img_io = io.BytesIO()
    pil_img.save(img_io, 'PNG')
    img_io.seek(0)

    return send_file(img_io, mimetype='image/png')

if __name__ == '__main__':
    app.run(debug=True)
