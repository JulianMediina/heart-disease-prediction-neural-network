# Predicción de Enfermedades Cardíacas con Redes Neuronales

Este proyecto utiliza redes neuronales para predecir el riesgo de enfermedades cardíacas a partir de un conjunto de datos de características de salud. Se ha implementado utilizando PyTorch y se han realizado mejoras en la arquitectura del modelo para optimizar la precisión de las predicciones.

## Tabla de Contenidos

- [Descripción](#descripción)
- [Tecnologías Utilizadas](#tecnologías-utilizadas)
- [Dataset](#dataset)
- [Instalación](#instalación)
- [Uso](#uso)
- [Resultados](#resultados)
- [Gráficos](#gráficos)

## Descripción

El objetivo de este proyecto es crear un modelo de red neuronal que pueda predecir el riesgo de enfermedades cardíacas basándose en características como la edad, el género, la presión arterial, el colesterol, entre otros. Se han implementado técnicas de mejora como la regularización y la detención temprana para evitar el sobreajuste.

## Tecnologías Utilizadas

- Python
- PyTorch
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn

## Dataset

El conjunto de datos utilizado es el **Heart Disease Dataset**. Este conjunto contiene información de salud de pacientes y se encuentra en el archivo `heart.csv`. 

## Instalación

Para configurar el entorno y las dependencias necesarias, puedes usar `pip`. Ejecuta el siguiente comando:
bash
`pip install -r requirements.txt`

## Uso
-Clona el repositorio en tu máquina local.
-Asegúrate de que las dependencias están instaladas.
-Ejecuta el archivo principal del proyecto:

`python heart_disease_prediction_nn.py`

## Resultados
Los resultados indican que el modelo tiene un buen rendimiento general, con alta precisión, recall y F1 Score, lo que sugiere que es capaz de clasificar correctamente tanto los casos positivos como negativos en el conjunto de datos.

## Gráficos

Este código genera los siguientes gráficos:

1. Matriz de Confusión:

Muestra la cantidad de verdaderos positivos, falsos positivos, verdaderos negativos y falsos negativos del modelo. Es útil para visualizar el rendimiento del modelo en la clasificación de las diferentes clases (presencia y ausencia de enfermedad cardíaca).

2. Pérdida de Entrenamiento y Prueba:

Un gráfico de líneas que representa cómo varía la pérdida durante el entrenamiento y la prueba a lo largo de las épocas. Ayuda a identificar si el modelo está aprendiendo correctamente (pérdida decreciente) y si hay sobreajuste (donde la pérdida de entrenamiento sigue disminuyendo mientras que la pérdida de prueba comienza a aumentar).

3. Distribución de Clases:

Un gráfico de barras que muestra la cantidad de ejemplos de cada clase en el conjunto de datos (casos negativos y positivos). Este gráfico es importante para verificar si hay un desequilibrio en las clases, lo que podría afectar el rendimiento del modelo.







