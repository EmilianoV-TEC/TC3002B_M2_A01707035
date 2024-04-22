import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os

from tensorflow.keras.models import load_model
from keras.preprocessing import image

path = "../data_unlabeled/"

test_dir = os.path.join(path, 'test')

# Cargamos el modelo guardado
autoencoder = load_model('models/Traffic_Sign_Autoencoder.h5')

test_image = image.load_img(os.path.join(test_dir, 'test_img_2.jpg'), target_size = (224, 224))
test_image = image.img_to_array(test_image)/255
test_image = np.expand_dims(test_image, axis = 0)

prediction = autoencoder(test_image)
prediction = np.squeeze(prediction)

plt.imshow(prediction)
plt.show()