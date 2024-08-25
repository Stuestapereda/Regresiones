import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#1 Analizar la correlación entre las variables

df = pd.read_csv("Advertising.csv",delimiter=",")

x1 = df["TV"]
x2 = df["Radio"]
x3 = df["Newspaper"]

y = df["Sales"]

for i in df:
    print(np.array(df[i]))

#Probando cambios
#Probando files 2
