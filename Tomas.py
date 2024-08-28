import pandas as pd
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt


# Cargamos el archivo 
df=pd.read_csv('Advertising.csv')
print(df.head())

# matriz de correlación para variables independientes
correlation_matrix = df[['TV', 'Radio','Newspaper','Sales']].corr(method=('pearson'))
print(correlation_matrix)

plt.figure(figsize=(8,6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
plt.title('Matriz de correlación')
plt.savefig(f'Matriz de correlación')
plt.show

# Función para calcular la regresíon lineal simple
def simple_linear_regression(X, y):

    # Calculate the mean of X and y
    mean_x = sum(X) / len(X)
    mean_y = sum(y) / len(y)

    # Calculate the slope (b1) and intercept (b0)
    numerator = sum((X - mean_x) * (y - mean_y))
    denominator = sum((X - mean_x) ** 2)
    b1 = numerator / denominator
    b0 = mean_y - b1 * mean_x

    # Predict y values using the linear regression equation y = b0 + b1 * X
    y_pred = [b0 + b1 * xi for xi in X]

    # Calculate Mean Absolute Error (MAE) and Mean Squared Error (MSE)
    mae = sum(abs(yi - ypi) for yi, ypi in zip(y, y_pred)) / len(y)
    mse = sum((yi - ypi) ** 2 for yi, ypi in zip(y, y_pred)) / len(y)

    return b0, b1, mae, mse, y_pred

# Extract TV advertising and Sales data
X = df['TV'].values
y = df['Sales'].values

# Perform simple linear regression
b0, b1, mae, mse, y_pred = simple_linear_regression(X, y)

# Print the results
print("")
print(f"Intercept (b0): {b0}")
print(f"Slope (b1): {b1}")
print(f"Mean Absolute Error (MAE): {mae}")
print(f"Mean Squared Error (MSE): {mse}")
print(f"Mean Squared Error (MSE): {y_pred}")
