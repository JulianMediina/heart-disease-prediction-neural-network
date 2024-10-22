# Importación de bibliotecas necesarias
import torch
from torch import nn, optim
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, classification_report, confusion_matrix, roc_curve,auc


# Cargar el dataset
# Se carga el archivo CSV de ataques cardíacos.
dataset = pd.read_csv('heart.csv')

# Preprocesamiento del dataset
# X son las características y y es la variable objetivo (resultado de ataque cardíaco o no).
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# Escalado de las características
# Se normalizan los valores de las características usando StandardScaler.
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Convertir los datos a tensores
# Los datos se convierten a tensores de PyTorch para entrenar el modelo.
X = torch.tensor(X, dtype=torch.float)
y = torch.tensor(y, dtype=torch.float).unsqueeze(dim=1)  # Se ajusta la dimensión de y

# División del dataset
# Se divide el conjunto de datos en entrenamiento (75%) y prueba (25%) con una semilla de aleatoriedad fija.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Definición del modelo de red neuronal con más capas y dropout
# Este modelo tiene capas adicionales y dropout para evitar sobreajuste.
class HeartAttackClassifier(nn.Module):
    def __init__(self):
        super(HeartAttackClassifier, self).__init__()
        self.linear = nn.Sequential(
            nn.Linear(13, 64),  # Capa de entrada con 13 características y 64 neuronas
            nn.ReLU(),          # Función de activación ReLU
            nn.Dropout(0.3),    # Dropout del 30%
            nn.Linear(64, 32),  # Capa intermedia de 64 a 32 neuronas
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, 1),   # Capa de salida con 1 neurona (para clasificación binaria)
            nn.Sigmoid()        # Función de activación Sigmoid para obtener probabilidades
        )

    def forward(self, x):
        return self.linear(x)

# Implementación de la función de pérdida Focal Loss
# Esta función de pérdida es útil cuando hay un desequilibrio en las clases.
class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        BCE_loss = nn.BCELoss()(inputs, targets)
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1-pt)**self.gamma * BCE_loss
        return torch.mean(F_loss) if self.reduction == 'mean' else torch.sum(F_loss)

# Función de entrenamiento con early stopping
# Esta función detiene el entrenamiento si no hay mejoras en la pérdida después de un número de épocas (patience).
def train_with_early_stopping(model, loss_fn, optimizer, patience=20):
    epochs = 1000  # Número máximo de épocas
    training_loss = []
    testing_loss = []
    best_test_loss = float('inf')
    early_stop_counter = 0

    for epoch in range(epochs):
        model.train()

        # Forward pass (paso hacia adelante)
        y_pred = model(X_train)
        loss = loss_fn(y_pred, y_train)

        # Backward pass (paso hacia atrás)
        optimizer.zero_grad()
        loss.backward()  # Calcula los gradientes
        optimizer.step()  # Actualiza los parámetros del modelo

        # Evaluación en el conjunto de prueba
        model.eval()
        with torch.inference_mode():
            test_pred = model(X_test)
            test_loss = loss_fn(test_pred, y_test)

        training_loss.append(loss.detach().numpy())
        testing_loss.append(test_loss.detach().numpy())

        # Lógica de early stopping
        if test_loss < best_test_loss:
            best_test_loss = test_loss
            early_stop_counter = 0  # Reinicia el contador
        else:
            early_stop_counter += 1  # Incrementa el contador de early stopping

        if early_stop_counter >= patience:
            print(f"Deteniendo temprano en la época {epoch}")
            break

        print(f"Época: {epoch} | Pérdida de entrenamiento: {loss} | Pérdida de prueba: {test_loss}")

    return training_loss, testing_loss

# Instanciar el modelo, optimizador y función de pérdida
model = HeartAttackClassifier()
optimizer = optim.Adam(model.parameters(), lr=0.001)
loss_fn = FocalLoss()

# Entrenar el modelo con early stopping
training_loss, testing_loss = train_with_early_stopping(model, loss_fn, optimizer)

# Función para hacer predicciones
def predict(model, X):
    model.eval()
    with torch.inference_mode():
        return model(X)

y_pred = predict(model, X_test).round()  # Redondeamos las probabilidades a 0 o 1 para obtener clases.

# Generar el reporte de evaluación
def generate_report(y_test, y_pred):
    print(f"Exactitud (Accuracy): {accuracy_score(y_test, y_pred)}")
    print(f"Recall: {recall_score(y_test, y_pred)}")
    print(f"Precisión: {precision_score(y_test, y_pred)}")
    print(f"F1 Score: {f1_score(y_test, y_pred)}")
    print(classification_report(y_test, y_pred))

    # Matriz de confusión
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='g', cmap='flare')

# Generar reporte del modelo entrenado
generate_report(y_test, y_pred)

# Guardar el modelo entrenado en un archivo
torch.save(model.state_dict(), 'model_heart_attack.pth')

# Visualización de las pérdidas de entrenamiento y prueba
plt.figure(figsize=(10, 5))
plt.plot(training_loss, label='Pérdida de entrenamiento')
plt.plot(testing_loss, label='Pérdida de prueba')
plt.title('Pérdida de entrenamiento y prueba a lo largo de las épocas')
plt.xlabel('Épocas')
plt.ylabel('Pérdida')
plt.legend()
plt.show()

#Grafico de Distribución de Clases en el Conjunto de Datos
def plot_class_distribution(y):
    plt.figure(figsize=(8, 6))
    
    # Convertir y a un tensor 1D
    y_1d = y.squeeze()  # Elimina dimensiones adicionales
    
    # Contar la cantidad de casos positivos y negativos
    class_counts = pd.Series(y_1d.numpy()).value_counts()
    
    # Crear el gráfico de barras
    sns.barplot(x=class_counts.index, y=class_counts.values, palette='flare')
    plt.title('Distribución de Clases en el Conjunto de Datos')
    plt.xlabel('Clase')
    plt.ylabel('Cantidad de Ejemplos')
    plt.xticks(ticks=[0, 1], labels=['Negativo (0)', 'Positivo (1)'])
    plt.ylim(0, max(class_counts.values) + 10)  # Para un mejor espaciado en el eje y
    plt.grid(axis='y')

    # Mostrar el valor de cada barra
    for index, value in enumerate(class_counts.values):
        plt.text(index, value + 1, str(value), ha='center')
    plt.show()

plot_class_distribution(y)




