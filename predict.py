import numpy as np
from keras.preprocessing import image
from tensorflow.keras.models import load_model
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import keras
import pathlib
import keras

# Charger le modèle pré-entraîné
model = load_model("C:/Users/X13/OneDrive/Bureau/CatDog/model_resnet.h5")

# Chemin de l'image à prédire
image_path = "1111.jpg"

# Charger et prétraiter l'image
img = image.load_img(image_path, target_size=(128, 128))
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)
img_array = img_array / 255.0  # Normaliser les valeurs des pixels

# Faire la prédiction
prediction = model.predict(img_array)

# Décoder la prédiction
if prediction[0][0] > prediction[0][1]:
    print("C'est un chat!")
else:
    print("C'est un chien!")
