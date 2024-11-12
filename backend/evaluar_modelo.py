from tensorflow.keras.models import load_model
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
import cv2
import numpy as np
import os
import csv

# Cargar el modelo entrenado
modelo = load_model("modelo_optimizado_v3.keras")

# Definir la ruta de la carpeta de pruebas y el archivo CSV con las etiquetas
ruta_pruebas = "C:/Users/Estefy Portillo/Desktop/Tes_2/backend/archive/jpeg224/test"  # Ruta de la carpeta de imágenes
archivo_csv = "C:/Users/Estefy Portillo/Desktop/Tes_2/backend/archive/jpeg224/train.csv"  # Archivo CSV con etiquetas

# Función para preprocesar las imágenes
def preprocesar_imagen(imagen, tamano_imagen=224):
    imagen = cv2.resize(imagen, (tamano_imagen, tamano_imagen))
    imagen = imagen / 255.0
    imagen = np.expand_dims(imagen, axis=0)
    return imagen

# Inicializar listas para etiquetas verdaderas y predicciones
etiquetas_verdaderas = []
predicciones = []
total_imagenes_procesadas = 0

# Leer las etiquetas verdaderas desde el archivo CSV
with open(archivo_csv, mode='r') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        nombre_imagen = row['image_name'] + ".jpg"  # Aseguramos el nombre de la imagen con la extensión
        etiqueta = int(row['benign_malign'])  # 0 para benigno, 1 para maligno

        # Construir la ruta completa de la imagen
        ruta_imagen = os.path.join(ruta_pruebas, nombre_imagen)
        imagen = cv2.imread(ruta_imagen)
        
        # Verificar si la imagen se cargó correctamente
        if imagen is None:
            print(f"Advertencia: No se pudo cargar la imagen '{nombre_imagen}'. Verifica la ruta o el formato.")
            continue
        
        # Preprocesar la imagen y hacer predicción
        imagen_preprocesada = preprocesar_imagen(imagen)
        prediccion = modelo.predict(imagen_preprocesada)
        etiqueta_predicha = np.argmax(prediccion)  # 0 para benigno, 1 para maligno
        
        # Almacenar la predicción y la etiqueta verdadera
        predicciones.append(etiqueta_predicha)
        etiquetas_verdaderas.append(etiqueta)
        total_imagenes_procesadas += 1

# Verificar si se procesaron imágenes
print(f"Total de imágenes procesadas: {total_imagenes_procesadas}")
print(f"Total de predicciones: {len(predicciones)}")
print(f"Total de etiquetas verdaderas: {len(etiquetas_verdaderas)}")

# Calcular métricas si hay predicciones y etiquetas verdaderas
if len(predicciones) > 0 and len(etiquetas_verdaderas) > 0:
    accuracy = accuracy_score(etiquetas_verdaderas, predicciones)
    precision = precision_score(etiquetas_verdaderas, predicciones, pos_label=1, zero_division=0)
    recall = recall_score(etiquetas_verdaderas, predicciones, pos_label=1, zero_division=0)
    f1 = f1_score(etiquetas_verdaderas, predicciones, pos_label=1, zero_division=0)

    # Mostrar resultados
    print("Exactitud (Accuracy):", accuracy)
    print("Precisión (Precision):", precision)
    print("Recuperación (Recall):", recall)
    print("F1 Score:", f1)
    print("\nInforme de clasificación:")
    print(classification_report(etiquetas_verdaderas, predicciones, target_names=["Benigno", "Maligno"], zero_division=0))
else:
    print("No se generaron predicciones o etiquetas verdaderas. Verifica la ruta de imágenes y el preprocesamiento.")
