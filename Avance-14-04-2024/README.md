# Actividad 2.1 Generación o selección del set de datos y Actividad 2.2 Preprocesado de los datos
### Emiliano Vásquez Olea - A01707035

Archivos: Traffic_Sign_Autoencoder_A01707035.py

Requisitos: Una versión actualizada de **Python 3** junto con las librerías **numpy**, **matplotlib**, **seaborn** y **tensorflow 2 (con keras)**

Para este avance no se está integrando una técnica de aprendizaje de máquina, debido a que por el momento solo se abarcan aspectos de obtención y procesado de los datos

## Dataset
El conjunto de dato utilizado para esta entrega es el de Traffic Sign Dataset - Classification (https://www.kaggle.com/datasets/ahemateja19bec1025/traffic-sign-dataset-classification). Este dataset está conformado por 6164 imagenes de señalamientos de tránsito agrupados en 58 clases o tipos distintos, estos datos pueden ser encontrados en la carpeta *data* en este repositorio.

Las imágenes utilizan el formato PNG y son redimensionadas para tener un formato 224 x 224 x 3. El dataset se encuentra dividido en conjuntos de prueba y entrenamiento así como por clases, aunque el objetivo es utilizar estos datos con una herramienta de aprendizaje no supervisado (Auto-encoders)
