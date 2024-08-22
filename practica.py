import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

df=pd.read_csv("Prueba.csv", delimiter=',')

y=np.array(df["peso"][0:-1])
x=np.array(df["altura"][0:-1])

num=0
den=0
for i in range(0,len(y)):
    num=num+(x[i]-np.mean(x))*(y[i]-np.mean(y))
    den=den+(x[i]-np.mean(x))**2

b1=num/den
b0=np.mean(y)-b1*np.mean(x)

def reg_lin(b1,b0,x):
    return x*b1+b0

x2=x
y2=np.zeros(len(y))

for i in range(0,len(y)):
    y2[i]=reg_lin(b1,b0,x[i])


plt.scatter(df["peso"],df["altura"])
plt.plot(y2,x2,c="red")
plt.show()

