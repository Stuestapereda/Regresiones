import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import combinations
from sklearn.model_selection import train_test_split

# Cargar el conjunto de datos
archivo = pd.read_csv("Advertising.csv")

# Definir las variables independientes
variables = ['TV', 'Radio', 'Newspaper']

# Dividir el conjunto de datos en conjuntos de entrenamiento y prueba (70% entrenamiento, 30% prueba)
entrenamiento, prueba = train_test_split(archivo, test_size=0.3, random_state=42)

# Matriz de correlación para las variables independientes
matriz_corr = entrenamiento[variables].corr()

# Graficar la matriz de correlación
plt.figure(figsize=(8, 6))
sns.heatmap(matriz_corr, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Matriz de Correlación para las Variables Independientes')
plt.show()

# Función para calcular el Factor de Inflación de Varianza (VIF) para un conjunto dado de variables
def calcular_vif(X):
    datos_vif = {}
    for i in range(len(X.columns)):
        X_i = X.iloc[:, i]
        X_sin_i = X.drop(X.columns[i], axis=1)
        R2_i = np.corrcoef(X_i, np.linalg.lstsq(X_sin_i, X_i, rcond=None)[0] @ X_sin_i.T)[0, 1] ** 2
        datos_vif[X.columns[i]] = 1 / (1 - R2_i)
    return datos_vif

# Función para calcular el R-cuadrado ajustado
def r_cuadrado_ajustado(r_cuadrado, n, p):
    return 1 - (1 - r_cuadrado) * (n - 1) / (n - p - 1)

# Preparar listas para almacenar resultados
resultados = []

# Realizar regresión para cada combinación de variables independientes
for i in range(1, len(variables) + 1):
    for combo in combinations(variables, i):
        X_entrenamiento = entrenamiento[list(combo)]
        y_entrenamiento = entrenamiento['Sales']
        X_prueba = prueba[list(combo)]
        y_prueba = prueba['Sales']
        
        # Añadir término de intercepto
        X_entrenamiento = np.c_[np.ones(X_entrenamiento.shape[0]), X_entrenamiento]
        X_prueba = np.c_[np.ones(X_prueba.shape[0]), X_prueba]
        
        # Calcular coeficientes utilizando MCO (Mínimos Cuadrados Ordinarios)
        beta = np.linalg.inv(X_entrenamiento.T @ X_entrenamiento) @ X_entrenamiento.T @ y_entrenamiento
        
        # Hacer predicciones
        y_pred = X_prueba @ beta
        
        # Calcular MSE y MAE
        mse = np.mean((y_prueba - y_pred) ** 2)
        mae = np.mean(np.abs(y_prueba - y_pred))
        
        # Calcular R-cuadrado
        ss_tot = np.sum((y_prueba - np.mean(y_prueba)) ** 2)
        ss_res = np.sum((y_prueba - y_pred) ** 2)
        r_cuadrado = 1 - (ss_res / ss_tot)
        
        # Calcular R-cuadrado ajustado
        r_cuadrado_aj = r_cuadrado_ajustado(r_cuadrado, len(y_prueba), len(combo))
        
        # Calcular VIF
        X_entrenamiento_sin_intercepto = entrenamiento[list(combo)]
        vif = calcular_vif(X_entrenamiento_sin_intercepto)
        
        # Almacenar resultados
        resultados.append({
            'Variables': combo,
            'Coeficientes': beta,
            'MSE': mse,
            'MAE': mae,
            'R-cuadrado': r_cuadrado,
            'R-cuadrado ajustado': r_cuadrado_aj,
            'VIF': vif,
            'y_pred': y_pred,  # Almacenar predicciones
            'y_prueba': y_prueba  # Almacenar valores reales
        })

# Mostrar resultados
for resultado in resultados:
    print(f"Variables: {resultado['Variables']}")
    print(f"Coeficientes: {resultado['Coeficientes']}")
    print(f"MSE: {resultado['MSE']:.2f}")
    print(f"MAE: {resultado['MAE']:.2f}")
    print(f"R-cuadrado: {resultado['R-cuadrado']:.2f}")
    print(f"R-cuadrado ajustado: {resultado['R-cuadrado ajustado']:.2f}")
    print(f"VIF: {resultado['VIF']}\n")

# Graficar predicciones vs valores reales para el modelo multivariado con las 3 variables
modelo_tres_variables = next((resultado for resultado in resultados if len(resultado['Variables']) == 3), None)

if modelo_tres_variables:
    plt.figure(figsize=(8, 6))
    plt.scatter(modelo_tres_variables['y_prueba'], modelo_tres_variables['y_pred'], alpha=0.7)
    plt.plot([min(modelo_tres_variables['y_prueba']), max(modelo_tres_variables['y_prueba'])],
             [min(modelo_tres_variables['y_prueba']), max(modelo_tres_variables['y_prueba'])],
             color='red', linestyle='--')
    plt.xlabel('Valores Reales')
    plt.ylabel('Predicciones')
    plt.title('Predicciones vs Valores Reales (Modelo con TV, Radio, y Newspaper)')
    plt.show()
