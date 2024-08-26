import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

# 1. Lectura y preparación de datos

# Leer el archivo CSV
data = pd.read_csv('german_credit.csv')

# Separar la variable objetivo (la primera columna) y las variables predictoras (todas las demás)
y = data.iloc[:, 0].values  # Variable objetivo
X = data.iloc[:, 1:].values  # Variables predictoras

# Dividir aleatoriamente los datos en conjuntos de entrenamiento (70%) y prueba (30%)
np.random.seed(42)  # Para asegurar que los resultados sean reproducibles
indices = np.random.permutation(len(X))  # Mezcla los índices de los datos
train_size = int(0.7 * len(X))  # Calcula el tamaño del conjunto de entrenamiento (70%)
train_indices, test_indices = indices[:train_size], indices[train_size:]  # Divide los índices en entrenamiento y prueba

X_train, X_test = X[train_indices], X[test_indices]  # Conjunto de entrenamiento y prueba para las variables predictoras
y_train, y_test = y[train_indices], y[test_indices]  # Conjunto de entrenamiento y prueba para la variable objetivo

# 2. Implementación del modelo de regresión logística

def sigmoid(z):
    """
    Función sigmoide: transforma la salida del modelo lineal en una probabilidad entre 0 y 1.
    """
    return 1 / (1 + np.exp(-z))

def predict_proba(X, beta):
    """
    Calcula la probabilidad predicha de la variable objetivo siendo 1 para un conjunto de datos X.
    
    X: matriz de características
    beta: vector de coeficientes del modelo
    """
    z = np.dot(X, beta)  # Producto punto entre X y beta
    return sigmoid(z)  # Aplicar la función sigmoide

# 3. Entrenamiento del modelo

def log_likelihood(X, y, beta):
    """
    Calcula la función de log-verosimilitud para la regresión logística.
    
    X: matriz de características
    y: vector de la variable objetivo
    beta: vector de coeficientes del modelo
    """
    z = np.dot(X, beta)  # Producto punto entre X y beta
    likelihood = np.sum(y * z - np.log(1 + np.exp(z)))  # Cálculo de la log-verosimilitud
    return likelihood

def gradient(X, y, beta):
    """
    Calcula el gradiente de la log-verosimilitud respecto a los coeficientes beta.
    
    X: matriz de características
    y: vector de la variable objetivo
    beta: vector de coeficientes del modelo
    """
    predictions = predict_proba(X, beta)  # Probabilidades predichas
    errors = y - predictions  # Diferencia entre la variable objetivo y las predicciones
    grad = np.dot(X.T, errors)  # Cálculo del gradiente
    return grad

def hessian(X, beta):
    """
    Calcula la matriz Hessiana (segunda derivada) de la log-verosimilitud.
    
    X: matriz de características
    beta: vector de coeficientes del modelo
    """
    predictions = predict_proba(X, beta)  # Probabilidades predichas
    diag = predictions * (1 - predictions)  # Vector diagonal para la matriz Hessiana
    H = np.dot(X.T, diag[:, np.newaxis] * X)  # Cálculo de la matriz Hessiana
    return H

def newton_raphson(X, y, max_iter=100, tol=1e-6):
    """
    Algoritmo de Newton-Raphson para optimizar los coeficientes beta.
    
    X: matriz de características
    y: vector de la variable objetivo
    max_iter: número máximo de iteraciones permitidas
    tol: tolerancia para la convergencia
    """
    beta = np.zeros(X.shape[1])  # Inicializar los coeficientes beta en cero
    for i in range(max_iter):
        grad = gradient(X, y, beta)  # Calcular el gradiente
        H = hessian(X, beta)  # Calcular la matriz Hessiana
        delta = np.linalg.solve(H, grad)  # Resolver el sistema lineal para encontrar la actualización de beta
        beta += delta  # Actualizar los coeficientes beta
        if np.linalg.norm(delta, 1) < tol:  # Verificar la convergencia
            break
    return beta

# Agregar una columna de 1s a X para incluir el término de intercepto en el modelo
X_train_intercept = np.hstack((np.ones((X_train.shape[0], 1)), X_train))
X_test_intercept = np.hstack((np.ones((X_test.shape[0], 1)), X_test))

# Entrenar el modelo usando el algoritmo de Newton-Raphson
beta_optimal = newton_raphson(X_train_intercept, y_train)

# 4. Predicción y evaluación

def predict(X, beta):
    """
    Predice la clase (0 o 1) para un conjunto de datos X basado en un umbral de 0.5.
    
    X: matriz de características
    beta: vector de coeficientes del modelo
    """
    return predict_proba(X, beta) >= 0.5

# Hacer predicciones en el conjunto de prueba
y_pred = predict(X_test_intercept, beta_optimal)

# Calcular la matriz de confusión y las métricas de rendimiento
cm = confusion_matrix(y_test, y_pred)  # Matriz de confusión
accuracy = accuracy_score(y_test, y_pred)  # Precisión
precision = precision_score(y_test, y_pred)  # Precisión
recall = recall_score(y_test, y_pred)  # Recall
f1 = f1_score(y_test, y_pred)  # F1-score

# Imprimir las métricas de rendimiento
print(f"Accuracy: {accuracy:.2f}")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1 Score: {f1:.2f}")

# 5. Visualización de resultados

# Generar la representación visual de la matriz de confusión
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')  # Crear un mapa de calor para la matriz de confusión
plt.xlabel('Predicción')
plt.ylabel('Real')
plt.title('Matriz de Confusión')
plt.show()
