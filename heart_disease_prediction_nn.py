# Importar librerías necesarias
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam

# Leer el archivo CSV
file_path = 'heart.csv'  # Cambia la ruta al archivo localmente
data = pd.read_csv(file_path)

# Mostrar las primeras filas del dataset
print("Primeras filas del dataset:")
print(data.head())

# Descripción estadística del dataset
print("\nDescripción estadística del dataset:")
print(data.describe())

# Información del dataset
print("\nInformación del dataset:")
print(data.info())

# Verificar valores nulos
print("\nValores nulos en el dataset:")
print(data.isnull().sum())

# Visualización de la distribución de la variable objetivo
plt.figure(figsize=(8, 6))
sns.countplot(x='output', data=data)
plt.title('Distribución de la variable objetivo (output)')
plt.xlabel('Enfermedad cardíaca (1: Sí, 0: No)')
plt.ylabel('Conteo')
plt.show()

# Visualizar la matriz de correlación
correlation_matrix = data.corr()
plt.figure(figsize=(12, 10))
sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='coolwarm')
plt.title('Matriz de Correlación')
plt.show()

# Separar las características (X) de la variable objetivo (y)
X = data.drop('output', axis=1)
y = data['output']

# Dividir los datos en entrenamiento (80%) y prueba (20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalizar las características numéricas
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Configurar la red neuronal
model = Sequential([
    Dense(2, activation='relu', input_shape=(X_train_scaled.shape[1],)),
    Dropout(0.2),  # Capa de Dropout
    Dense(1, activation='relu'),
    Dropout(0.2),  # Capa de Dropout
    Dense(1, activation='sigmoid')  # Salida binaria (enfermedad cardíaca o no)
])

# Compilar el modelo
optimizer = Adam(learning_rate=0.001)  # Ajustar la tasa de aprendizaje
model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

# Configurar Early Stopping
#early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# Entrenar la red neuronal
history = model.fit(X_train_scaled, y_train, epochs=200, batch_size=16, validation_split=0.2, verbose=1)

# Evaluar el modelo con los datos de prueba
test_loss, test_accuracy = model.evaluate(X_test_scaled, y_test)

print(f'\nPérdida en el conjunto de prueba: {test_loss}')
print(f'Precisión en el conjunto de prueba: {test_accuracy}')

# Graficar la pérdida y precisión durante el entrenamiento
plt.figure(figsize=(12, 5))

# Pérdida
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Pérdida de entrenamiento')
plt.plot(history.history['val_loss'], label='Pérdida de validación')
plt.title('Pérdida durante el entrenamiento')
plt.xlabel('Épocas')
plt.ylabel('Pérdida')
plt.legend()

# Precisión
plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Precisión de entrenamiento')
plt.plot(history.history['val_accuracy'], label='Precisión de validación')
plt.title('Precisión durante el entrenamiento')
plt.xlabel('Épocas')
plt.ylabel('Precisión')
plt.legend()

plt.tight_layout()
plt.show()
