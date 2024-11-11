import os
import cv2
import numpy as np
from tqdm import tqdm
from tensorflow.keras.applications import EfficientNetB3
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split

# Directorio de datos
DATADIR = "C:/Users/Estefy Portillo/Desktop/Tes_1/backend/archive/jpeg224"
CATEGORIAS = ['test', 'train']

# Función para cargar datos
def cargar_datos(tamano_imagen=224):
    datos, etiquetas = [], []
    for categoria in CATEGORIAS:
        ruta_categoria = os.path.join(DATADIR, categoria)
        valor = CATEGORIAS.index(categoria)

        for imagen_nombre in tqdm(os.listdir(ruta_categoria)):
            imagen_ruta = os.path.join(ruta_categoria, imagen_nombre)
            imagen = cv2.imread(imagen_ruta)

            if imagen is not None:
                imagen = cv2.resize(imagen, (tamano_imagen, tamano_imagen))  # Redimensionar
                datos.append(imagen)
                etiquetas.append(valor)
            else:
                print(f"Error al cargar la imagen: {imagen_ruta}")
    
    return np.array(datos), np.array(etiquetas)

# Cargar datos y dividir en entrenamiento y validación
datos, etiquetas = cargar_datos()
X_train, X_val, y_train, y_val = train_test_split(datos, etiquetas, test_size=0.2, random_state=42)

# Generador de aumento de datos avanzado
data_gen = ImageDataGenerator(
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.3,
    brightness_range=[0.8, 1.2],
    horizontal_flip=True,
    fill_mode="nearest"
)

# Definición del modelo optimizado con EfficientNetB3
def crear_modelo_optimizado():
    # Cargar EfficientNetB3 preentrenado en ImageNet, sin la capa superior
    base_model = EfficientNetB3(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    
    # Descongelar parcialmente para ajuste fino
    base_model.trainable = True
    for layer in base_model.layers[:-20]:  # Mantener congeladas todas menos las últimas 20 capas
        layer.trainable = False

    # Añadir capas superiores al modelo
    modelo = Sequential([
        base_model,
        Flatten(),
        Dense(512, activation='relu'),  # Capa densa para refinar las características
        Dropout(0.5),  # Dropout ajustado para reducir sobreajuste
        Dense(2, activation='softmax')  # Capa de salida para clasificación binaria
    ])

    # Compilar el modelo con Adam y tasa de aprendizaje baja
    modelo.compile(optimizer=Adam(learning_rate=0.00001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    return modelo

# Crear y entrenar el modelo
modelo_optimizado = crear_modelo_optimizado() 
history = modelo_optimizado.fit(data_gen.flow(X_train, y_train, batch_size=32),
                                validation_data=(X_val, y_val),
                                epochs=20)

# Guardar el modelo optimizado
modelo_optimizado.save("modelo_optimizado_v3.keras")    
