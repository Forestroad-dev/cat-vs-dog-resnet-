import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img
from keras.models import Sequential, Model
from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense, BatchNormalization, Add, Input, Activation
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
import matplotlib.pyplot as plt
import os
from keras.preprocessing import image

def is_valid_image(img_path):
    try:
        # Tentative d'ouverture de l'image
        img = image.load_img(img_path)
        return True
    except:
        # Si une erreur se produit, l'image est considérée comme invalide
        return False

# Chemins vers les répertoires contenant les images de chats et de chiens
cats_dir = "D:/M2IABD/computer-vision/images/Cat/"
dogs_dir = "D:/M2IABD/computer-vision/images/Dog/"

# Liste des noms de fichiers des images de chats et de chiens
cat_filenames = [os.path.join(cats_dir, filename) for filename in os.listdir(cats_dir) if is_valid_image(os.path.join(cats_dir, filename))]
dog_filenames = [os.path.join(dogs_dir, filename) for filename in os.listdir(dogs_dir) if is_valid_image(os.path.join(dogs_dir, filename))]

# Création du DataFrame avec les noms de fichiers et les étiquettes de classe
filenames = cat_filenames + dog_filenames
categories = ['cat'] * len(cat_filenames) + ['dog'] * len(dog_filenames)

df = pd.DataFrame({
    'filename': filenames,
    'category': categories
})

# Définition des dimensions des images
Image_Width = 128
Image_Height = 128
Image_Size = (Image_Width, Image_Height)
Image_Channels = 3

# Fonction pour construire un bloc résiduel
def residual_block(x, filters, conv_num=3, activation='relu'):
    # Shortcut
    s = Conv2D(filters, 1, padding='same')(x)
    for i in range(conv_num - 1):
        x = Conv2D(filters, 3, padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation(activation)(x)
    x = Conv2D(filters, 3, padding='same')(x)
    x = BatchNormalization()(x)
    # Ajout de la shortcut à la sortie du dernier conv layer
    x = Add()([x, s])
    x = Activation(activation)(x)
    return MaxPooling2D(pool_size=(2, 2))(x)

# Création du modèle ResNet simplifié
inputs = Input(shape=(Image_Width, Image_Height, Image_Channels))

# Première convolution et max pooling
x = Conv2D(32, 3, activation='relu', padding='same')(inputs)
x = MaxPooling2D(2)(x)

# Ajout de 3 blocs résiduels
x = residual_block(x, 64)
x = residual_block(x, 128)
x = residual_block(x, 256)

# Flatten et fully connected layers
x = Flatten()(x)
x = Dense(512, activation='relu')(x)
x = Dropout(0.5)(x)
outputs = Dense(2, activation='softmax')(x)

# Création du modèle
model = Model(inputs, outputs)

# Compilation du modèle
model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

# Affichage de la structure du modèle
model.summary()

# Définition des callbacks pour l'entraînement du modèle
earlystop = EarlyStopping(patience=10)
learning_rate_reduction = ReduceLROnPlateau(monitor='val_accuracy', patience=2, verbose=1, factor=0.5, min_lr=0.00001)
callbacks = [earlystop, learning_rate_reduction]

# Préparation des générateurs de données
batch_size = 15
train_datagen = ImageDataGenerator(rotation_range=15, rescale=1./255, shear_range=0.1, zoom_range=0.2,
                                   horizontal_flip=True, width_shift_range=0.1, height_shift_range=0.1, validation_split=0.2
                                   )
train_generator = train_datagen.flow_from_dataframe(df, directory=None, x_col='filename', y_col='category',
                                                    target_size=Image_Size, class_mode='categorical',
                                                    batch_size=batch_size, subset='training',  validate_filenames=False)

# Création du générateur de données pour la validation
validation_generator = train_datagen.flow_from_dataframe(
    df,
    x_col='filename',
    y_col='category',
    target_size=Image_Size,
    class_mode='categorical',
    batch_size=batch_size,
    subset='validation',  # Utilisation uniquement des données de validation
    validate_filenames=False,  # Ignorer les fichiers invalides
   )

# Entraînement du modèle
epochs = 10
history = model.fit(train_generator, epochs=epochs, callbacks=callbacks)

# Sauvegarde du modèle
model.save("model_resnet.h5")
