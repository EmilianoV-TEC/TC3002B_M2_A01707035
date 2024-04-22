# Actividad 2.3 Implementación de Modelo y Actividad 2.4 Evaluación inicial del modelo
### Emiliano Vásquez Olea - A01707035

Archivos: Entrenamiento_Autoencoder_A01707035.py, Despliegue_Autoencoder_A01707035.py, Traffic_Sign_Autoencoder.h5

Requisitos: Una versión actualizada de **Python 3** junto con las librerías **numpy**, **matplotlib**, **seaborn** y **tensorflow 2 (con keras)**

La ejecución del código para el entrenamiento del modelo puede tardar varios minutos.

## Dataset
El conjunto de dato utilizado para esta entrega es el de Traffic Sign Dataset - Classification (https://www.kaggle.com/datasets/ahemateja19bec1025/traffic-sign-dataset-classification). Este dataset está conformado por 6164 imagenes de señalamientos de tránsito agrupados en 58 clases o tipos distintos, estos datos pueden ser encontrados en la carpeta *data* en este repositorio.

Las imágenes utilizan el formato PNG y son redimensionadas para tener un formato 224 x 224 x 3. El dataset se encuentra dividido en conjuntos de prueba y entrenamiento así como por clases, aunque el objetivo es utilizar estos datos con una herramienta de aprendizaje no supervisado (Auto-encoders)

## Técnica Implementada
La técnica de aprendizaje automático utilizada para el conjunto de datos es un Autoencoder, que pertenece al área de aprendizaje no supervisado. Un autoencoder consiste de una red neuronal conformada por dos
secciones principales, un encoder y un decoder, que se encargan de codificar/comprimir y decodificar una imagen respectivamente dentro de la red. Los componentes de esta red permiten representar imágenes de formas más compactas así como recounstruir imágenes que se encuentren en un formato compacto de igual forma.

El encoder está conformado por una serie de capas de convolución y Max-pooling, mientras que el decoder incluye capas de convolución y Up-sampling para regresar al formato original de la imagen. Para medir
el rendimiento del modelo durante el proceso de entrenamiento se utiliza la función de pérdida a lo largo de las épocas, debido a que pueden verse limitadas las métricas de precisión al ser una técnica de aprendizaje no supervisado. El modelo es almacenado en la carpeta **models** posterior al entrenamiento.

Un artículo de investigación en el que se integran redes neuronales como Autoencoders es el de 'Nonlinear principal component analysis using autoassociative neural networks' por Mark A. Kramer. En este artículo se utiliza esta técnica para el proceso de análisis de componentes principales:

Kramer, M.A. (1991), Nonlinear principal component analysis using autoassociative neural networks. AIChE J., 37: 233-243. https://doi.org/10.1002/aic.690370209