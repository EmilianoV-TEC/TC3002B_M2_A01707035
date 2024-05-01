import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Flatten, Reshape, Conv2D, MaxPooling2D, InputLayer, UpSampling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator

path = "../data_unlabeled/"

train_dir = os.path.join(path,'train')

train_datagen = ImageDataGenerator(
                rescale = 1./255,
                #rotation_range = 40,
				#width_shift_range = 0.1,
				#height_shift_range = 0.1,
                #horizontal_flip = True
                )

train_generator = train_datagen.flow_from_directory(
							train_dir,
							target_size = (224, 224),
							batch_size = 8,
							class_mode ='input',
                            seed = 20
                            )

# Definición del modelo
encoder = Sequential([
    InputLayer(input_shape = (224, 224, 3)),
    Conv2D(32, (3, 3), activation="relu", padding='same'),
    Conv2D(16, (3, 3), activation="relu", padding='same'),
    MaxPooling2D((2, 2), padding='same'),
    Conv2D(8, (3, 3), activation="relu", padding='same'),
    MaxPooling2D((2, 2), padding='same'),
    Conv2D(4, (3, 3), activation="relu", padding='same'),
    MaxPooling2D((2, 2), padding='same'),
])

decoder = Sequential([
    InputLayer(input_shape = (28, 28, 4)),
    Conv2D(4, (3, 3), activation="relu", padding='same'),
    UpSampling2D(size = (2, 2)),
    Conv2D(8, (3, 3), activation="relu", padding='same'),
    UpSampling2D(size = (2, 2)),
    Conv2D(16, (3, 3), activation="relu", padding = 'same'),
    UpSampling2D(size = (2, 2)),
    Conv2D(32, (3, 3), activation="relu", padding = 'same'),
    Conv2D(3, (3, 3), padding = 'same')
    ])

autoencoder = Model(inputs = encoder.inputs, outputs = decoder(encoder.outputs))

# Compilar el modelo
autoencoder.compile(optimizer = 'adam', loss = 'binary_crossentropy')

# Entrenar el modelo
history = autoencoder.fit(train_generator, steps_per_epoch = 100, epochs = 15)

# Guardar el modelo
autoencoder.save('models/Traffic_Sign_Autoencoder_extraconvolution.h5')

# Mostrar el historial de la función de pérdida a lo largo del entrenamiento
history_df = pd.DataFrame(history.history)

sns.lineplot(data = history_df[['loss']])
plt.show()