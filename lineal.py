import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from scipy import stats
from sklearn.metrics import mean_absolute_error, mean_squared_error
import seaborn as sns



#Funciones
def coeficientes(x1,y):
    """
    Calcula los coeficientes de la regresión lineal.
    """
    
    num=0
    den=0
    for i in range(0,len(x1)):
        num+=(x1[i]-np.mean(x1))*(y[i]-np.mean(y))
        den+=(x1[i]-np.mean(x1))**2
    b1=num/den
    b0=np.mean(y)-b1*np.mean(x1)

    return b1,b0

def plotear_regresion(x,y,y1,d):
    """
    Grafica la regresión lineal y guarda el gráfico en un archivo.
    """

    plt.figure(figsize=(10, 6))
    plt.plot(x,y1,"red",label=f"{d}")
    plt.scatter(x,y,label="Datos originales")
    plt.xlabel(f"{d}")
    plt.ylabel("Sales")
    plt.savefig(f'Regresión_lineal_{d}.png')
    plt.legend()
    #plt.show()

def SE(x,y,yi):
    """
    Calcula el Error Estándar del coeficiente de regresión.
    """

    y=np.array(y)
    yi=np.array(yi)
    x=np.array(x)
    e=y-yi

    SSR=0
    for i in range(0,len(y)):
        SSR+=e[i]**2

    s=(SSR/(len(y)-2))**0.5

    den=0
    for i in range(0,len(y)):
        den+=(x[i]-np.mean(x))**2
    den=den**0.5

    return s/den

def coef_Det(y,yi):
    """
    Calcula el coeficiente de determinación R^2.
    """

    num=0
    den=0
    for i in range(0,len(y)):
        num+=(yi[i]-y[i])**2
        den+=(y[i]-np.mean(y))**2

    return 1-num/den

#1 Analizar la correlación entre las variables -------------

df = pd.read_csv("Advertising.csv",delimiter=",")

# Calcular la matriz de correlación
correlation_matrix = df[['TV', 'Radio', 'Newspaper', 'Sales']].corr(method=("pearson"))

# Graficar el heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
plt.title('Matriz de Correlación')
plt.savefig(f'Matriz de Correlación')
#plt.show()

#2 Regresión lineal ----------------------------------------
y=df["Sales"][:-40]

#Analizando la relación entre "sales" y la publicidad por TV
x1=df["TV"][:-40]
b1_1,b0_1=coeficientes(x1,y)
y1=[b0_1+b1_1*i for i in x1]
plotear_regresion(x1,y,y1,"TV")
print(f"b0={b0_1}, b1={b1_1}")

#Analizando la relación entre "sales" y la publicidad por TV
x2=df["Radio"][:-40]
b1_2,b0_2=coeficientes(x2,y)
y2=[b0_2+b1_2*i for i in x2]
plotear_regresion(x2,y,y2,"Radio")
print(f"b0={b0_2}, b1={b1_2}")

#Analizando la relación entre "sales" y la publicidad por TV
x3=df["Newspaper"][:-40]
b1_3,b0_3=coeficientes(x3,y)
y3=[b0_3+b1_3*i for i in x3]
plotear_regresion(x3,y,y3,"Newspaper")
print(f"b0={b0_3}, b1={b1_3}")

#3 Realizando diagnóstico ----------------------------------

"""
t de student:
H0: b1=0

Conclusión:
Dado que todos los pvalue son menores que 0.05 se rechaza la hipotesis nula y se concluye que
los coeficientes son estadísticamente diferentes de 0 y que la variable predictora tiene un 
efecto significativo sobre la variable dependiente

"""
#Calculando SE, el estadístico t y el p value el para TV vs Sales
SE_1=SE(x1,y,y1)
t1 = b1_1/SE_1
p_value1 = 2 * (1 - stats.t.cdf(abs(t1), len(x1)))

#Calculando SE para Radio vs Sales
SE_2=SE(x2,y,y2)
t2 = b1_2/SE_2
p_value2 = 2 * (1 - stats.t.cdf(abs(t2), len(x2)))

#Calculando SE para Newspaper vs Sales
SE_3=SE(x3,y,y3)
t3 = b1_3/SE_3
p_value3 = 2 * (1 - stats.t.cdf(abs(t3), len(x3)))

print(f"p-value Radio: {p_value2}")
print(f"p-value TV: {p_value1}")
print(f"p-value Newspaper: {p_value3}")

"""
Coeficiente de determinación (R^2)
"""

c1 = coef_Det(y, y1)
c2 = coef_Det(y, y2)
c3 = coef_Det(y, y3)

print(f"R^2 TV: {c1}")
print(f"R^2 Radio: {c2}")
print(f"R^2 Newspaper: {c3}")

#4 Calcular las métricas MAE y MSE -----------------------------


# Calcular MAE (Mean Absolute Error) para Tv
mae1 = mean_absolute_error(y, y1)
print(f"Mean Absolute Error (MAE): {mae1} para Tv vs Sales")
# Calcular MSE (Mean Squared Error) para Tv
mse1 = mean_squared_error(y, y1)
print(f"Mean Squared Error (MSE): {mse1} para Tv vs Sales")
# Calcular MAE (Mean Absolute Error) para Radio
mae2 = mean_absolute_error(y, y2)
print(f"Mean Absolute Error (MAE): {mae2} para Radio vs Sales")
# Calcular MSE (Mean Squared Error) para Radio
mse2 = mean_squared_error(y, y2)
print(f"Mean Squared Error (MSE): {mse2} para radio vs Sales")
# Calcular MAE (Mean Absolute Error) para Newspaper
mae3 = mean_absolute_error(y, y3)
print(f"Mean Absolute Error (MAE): {mae3} para newspaper vs Sales")
# Calcular MSE (Mean Squared Error) para Newspaper
mse3 = mean_squared_error(y, y3)
print(f"Mean Squared Error (MSE): {mse3} para newspaper vs Sales")



"""
#3 Regresión Multiple
Y=np.array(y)
X=np.array([np.array(x1), np.array(x2), np.array(x3)]).T
b=np.dot(np.dot((np.dot(X.T,X))**(-1),X.T),Y)

b0=b[0]
b1=b[1]
b2=b[2]
"""