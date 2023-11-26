import comet_ml
comet_api_key = 'XnWyxH3bAdQko2SPq2anGBzVq'
comet_project_name = 'Reconocimiento facial'
experiment = comet_ml.Experiment(api_key=comet_api_key, project_name=comet_project_name)

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten,Conv2D, Dropout, Activation, MaxPooling2D
import os
import random
from tensorflow.keras.preprocessing import image
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.utils import image_dataset_from_directory

train_dataset = image_dataset_from_directory(
    "data/train",
    image_size=(192, 192),
    batch_size=5)
test_dataset = image_dataset_from_directory(
    "data/test",
    image_size=(192, 192),
    batch_size=5)

# Cargar el modelo preentrenado
modelo_preentrenado = tf.keras.models.load_model('modelo_ident_celebA.h5')

# Congelar los pesos de todas las capas excepto la última
for capa in modelo_preentrenado.layers[:-1]:
    capa.trainable = False

# Crear un nuevo modelo para la identificación facial
modelo_rostro = Sequential()

# Agregar todas las capas excepto la última al nuevo modelo
for capa in modelo_preentrenado.layers[:-1]:
    modelo_rostro.add(capa)


from tensorflow.keras import layers


data_augmentation = keras.Sequential(
    [ layers.RandomFlip("horizontal"),
      layers.RandomRotation(0.1),
      layers.RandomZoom(0.2)
      ]
)





modelo_rostro.add(Dense(64, activation='relu', name='mi_capa_dense'))
modelo_rostro.add(Dense(64, activation='relu', name='mi_capa_dense2'))
modelo_rostro.add(Dense(64, activation='relu', name='mi_capa_dense3'))
modelo_rostro.add(Dense(1, activation='sigmoid', name='mi_capa_salida'))
modelo_rostro.compile(loss="binary_crossentropy", optimizer="rmsprop", metrics=["accuracy"])




history = modelo_rostro.fit(
    train_dataset,
    epochs=10,
    validation_data=test_dataset,
)



print(history.history.keys())
for key in history.history:
    experiment.log_metric(f"{key}", history.history[key][-1])

# Registra el modelo en Comet ML
experiment.log_remote_model(model_name=modelo_rostro, uri="modelo_rostro")



# Se realizan pruebas para saber si efectivamente detecta que es mi rostro o no
directorio_prueba_1 = 'img_align_celeba/'

archivos_en_directorio_1 = os.listdir(directorio_prueba_1)

# Seleccionar 10 imágenes al azar
imagenes_de_prueba = random.sample(archivos_en_directorio_1, 10)

fig, axs = plt.subplots(2, 5, figsize=(15, 6))
axs = axs.ravel()

for i, imagen_azar in enumerate(imagenes_de_prueba):
    ruta_imagen_prueba = os.path.join(directorio_prueba_1, imagen_azar)
    imagen_prueba = image.load_img(ruta_imagen_prueba, target_size=(192, 192))
    imagen_prueba_array = image.img_to_array(imagen_prueba)
    imagen_prueba_array = np.expand_dims(imagen_prueba_array, axis=0)
    imagen_prueba_array /= 255.0  # Normalizar la imagen al rango [0, 1]

    prediccion = modelo_rostro.predict(imagen_prueba_array)

    # La salida de la red será un valor entre 0 y 1
    umbral = 0.5
    resultado = "No es tu rostro" if prediccion[0][0] > umbral else "Es tu rostro"

    axs[i].imshow(imagen_prueba)
    axs[i].set_title(f'Predicción: {prediccion[0][0]}\n{resultado}')
    axs[i].axis('off')

plt.show()



directorio_prueba_2 = 'data/test/Mi rostro/'

archivos_en_directorio_2 = os.listdir(directorio_prueba_2)
imagenes_de_prueba_2 = random.sample(archivos_en_directorio_2, 10)

fig, axs = plt.subplots(2, 5, figsize=(15, 6))
axs = axs.ravel()

for i, imagen_azar in enumerate(imagenes_de_prueba_2):
    ruta_imagen_prueba2 = os.path.join(directorio_prueba_2, imagen_azar)
    imagen_prueba2 = image.load_img(ruta_imagen_prueba2, target_size=(192, 192))
    imagen_prueba_array2 = image.img_to_array(imagen_prueba2)
    imagen_prueba_array2 = np.expand_dims(imagen_prueba_array2, axis=0)
    imagen_prueba_array2 /= 255.0

    prediccion2 = modelo_rostro.predict(imagen_prueba_array2)

    umbral = 0.5
    resultado2 = "No es tu rostro" if prediccion2[0][0] > umbral else "Es tu rostro"

    axs[i].imshow(imagen_prueba2)
    axs[i].set_title(f'Predicción: {prediccion2[0][0]}\n{resultado2}')
    axs[i].axis('off')

plt.show()

experiment.end()


modelo_rostro.save('modelo_ident_rostro.h5')

