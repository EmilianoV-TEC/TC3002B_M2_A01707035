import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os

from tensorflow.keras.preprocessing.image import ImageDataGenerator

path = "../data/"

train_dir = os.path.join(path,'train')
test_dir = os.path.join(path, 'test')

train_datagen = ImageDataGenerator(
                1./255,
                #rotation_range = 40,
				#width_shift_range = 0.1,
				#height_shift_range = 0.1,
                #horizontal_flip = True
                )

train_generator = train_datagen.flow_from_directory(
							train_dir,
							target_size = (224, 224),
							batch_size = 20,
							class_mode ='categorical')

fig, ax = plt.subplots(1, 5, figsize=(9, 6))

# Mostrar algunas imagenes de ejemplo
for i in range(5):
    batch = next(train_generator)

    example_image = batch[0][i]
    ax[i].imshow(example_image.astype('uint8'))

plt.show()