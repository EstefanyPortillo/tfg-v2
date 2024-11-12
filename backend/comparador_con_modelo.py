import cv2
import numpy as np
from tensorflow.keras.models import load_model
from app_with_comparison_and_prediction import comparar_y_predecir_imagenes 

# Ruta de tu modelo preentrenado
modelo_ruta = "modelo_optimizado_v3.keras"
modelo = load_model(modelo_ruta)

# Rutas de las imágenes que deseas comparar
ruta_imagen1 = r"C:\Users\Estefy Portillo\Desktop\TFG\archive\jpeg224\train\ISIC_0086349.jpg"
ruta_imagen2 = r"C:\Users\Estefy Portillo\Desktop\TFG\archive\jpeg224\train\ISIC_0086462.jpg"


# Llamar a la función para comparar y predecir
comparar_y_predecir_imagenes(ruta_imagen1, ruta_imagen2, modelo_ruta)
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from skimage.metrics import structural_similarity as ssim
from matplotlib import pyplot as plt

# Cargar el modelo preentrenado
modelo = load_model("modelo_optimizado_v3.keras")  # Cambia esto por el nombre correcto de tu modelo

# Función para preprocesar la imagen para el modelo
def preprocesar_imagen(imagen, tamano_imagen=224):
    imagen = cv2.resize(imagen, (tamano_imagen, tamano_imagen))
    imagen = imagen / 255.0  # Normalización
    imagen = np.expand_dims(imagen, axis=0)  # Expande la dimensión para el modelo
    return imagen

# Función para calcular el porcentaje de similitud
def calcular_porcentaje_similitud(imagen1, imagen2):
    imagen1_gray = cv2.cvtColor(imagen1, cv2.COLOR_BGR2GRAY)
    imagen2_gray = cv2.cvtColor(imagen2, cv2.COLOR_BGR2GRAY)
    score, _ = ssim(imagen1_gray, imagen2_gray, full=True)
    porcentaje_similitud = score * 100
    print(f"Porcentaje de similitud (SSIM): {porcentaje_similitud:.2f}%")
    return porcentaje_similitud

# Función para hacer predicción en cada imagen
def predecir_imagen(modelo, imagen):
    imagen_preprocesada = preprocesar_imagen(imagen)
    prediccion = modelo.predict(imagen_preprocesada)
    clase = "Lesión benigna" if np.argmax(prediccion, axis=1)[0] == 0 else "Lesión maligna"
    print(f"Predicción del modelo: {clase}")
    return clase

# Función para extraer y mostrar histogramas de color
def mostrar_histogramas_color(imagen, titulo="Histograma"):
    color = ('b', 'g', 'r')
    for i, col in enumerate(color):
        hist = cv2.calcHist([imagen], [i], None, [256], [0, 256])
        plt.plot(hist, color=col)
        plt.xlim([0, 256])
    plt.title(titulo)
    plt.show()

# Función para extraer bordes usando Canny
def extraer_bordes(imagen):
    imagen_gray = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
    bordes = cv2.Canny(imagen_gray, 100, 200)
    plt.imshow(bordes, cmap='gray')
    plt.title("Bordes detectados")
    plt.show()
    return bordes

# Cargar las imágenes
ruta_imagen1 = r"C:\Users\Estefy Portillo\Desktop\TFG\archive\jpeg224\train\ISIC_0086349.jpg"
ruta_imagen2 = r"C:\Users\Estefy Portillo\Desktop\TFG\archive\jpeg224\train\ISIC_0086462.jpg"
imagen1 = cv2.imread(ruta_imagen1)
imagen2 = cv2.imread(ruta_imagen2)

# Verificar que las imágenes se cargaron correctamente
if imagen1 is None or imagen2 is None:
    print("Error al cargar una o ambas imágenes.")
else:
    # Calcular y mostrar el porcentaje de similitud
    calcular_porcentaje_similitud(imagen1, imagen2)
    
    # Obtener y mostrar predicción para cada imagen
    print("Predicción - Imagen 1:")
    prediccion1 = predecir_imagen(modelo, imagen1)
    
    print("Predicción - Imagen 2:")
    prediccion2 = predecir_imagen(modelo, imagen2)
    
    # Comparar si ambas predicciones coinciden
    if prediccion1 == prediccion2:
        print("Ambas imágenes presentan el mismo tipo de lesión según el modelo.")
    else:
        print("Las imágenes presentan diferentes tipos de lesión según el modelo.")
    
    # Mostrar histogramas de color para ambas imágenes
    print("Histograma de color - Imagen 1:")
    mostrar_histogramas_color(imagen1, "Imagen 1")
    
    print("Histograma de color - Imagen 2:")
    mostrar_histogramas_color(imagen2, "Imagen 2")
    
    # Extraer y mostrar bordes para ambas imágenes
    print("Bordes - Imagen 1:")
    extraer_bordes(imagen1)
    
    print("Bordes - Imagen 2:")
    extraer_bordes(imagen2)
