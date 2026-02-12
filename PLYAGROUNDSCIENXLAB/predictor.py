#importar el archivo acientes csv a un dafaframe
#Cargar el archivo pacientes.csv, escalar las columnas edad y colesterol, y luego convertirlo al formato solicitado.
import pandas as pd
from sklearn.impute import SimpleImputer
import math
import matplotlib.pyplot as plt
import joblib

# 1. Cargar el archivo CSV
df = pd.read_csv('pacientes.csv')
#Realizar un grafico de dispersión entre edad y colesterol, coloreando los puntos según el problema_cardiaco
#imprimir el maximo y el minimo de la edad y del colesterol
print("Edad - Max:", df['edad'].max(), "Min:", df['edad'].min())
print("Colesterol - Max:", df['colesterol'].max(), "Min:", df['colesterol'].min())

#imprimir una grafica para ver la ditribucion de los puntos
plt.scatter(df['edad'], df['colesterol'], c=df['problema_cardiaco'], cmap='viridis')
plt.xlabel('Edad')
plt.ylabel('Colesterol')
plt.title('Dispersión entre Edad y Colesterol')
plt.colorbar(label='Problema Cardiaco')
plt.show()
                    
#Cargar el modelo de escalado de joblib del archivo scaler.joblib

scaler = joblib.load('scaler.jb')

#Con esta función ce una red neuronal se va a predecir si va a tener problemas cardiacos o no, se va a usar el modelo de escalado para escalar las columnas edad y colesterol 
def predecir_problema_cardiaco(edad, colesterol):
    # Escalar las columnas X1,x2 usando el modelo de escalado cargado
    datos_transformados = scaler.transform([[edad, colesterol]])
    X1 = datos_transformados[0, 0]*2
    X2 = datos_transformados[0, 1]*2
    print("Datos escalados - Edad:", X1, "Colesterol:", X2)
    # Convertir al formato solicitado
    """Compute a forward pass of the network."""
    a1 = 1 / (1 + math.exp(-(-2.0 + (2.3 * X1) + (-1.9 * X2))))
    a2 = 1 / (1 + math.exp(-(-6.0 + (1.3 * X1) + (-2.1 * X2))))
    a3 = 1 / (1 + math.exp(-(2.5 + (2.2 * X1) + (8.7 * X2))))
    a4 = 1 / (1 + math.exp(-(3.3 + (0.13 * X1) + (-2.0 * X2))))
    a5 = 1 / (1 + math.exp(-(1.8 + (-0.43 * X1) + (-2.5 * X2))))
    a6 = 1 / (1 + math.exp(-(1.7 + (-0.88 * X1) + (0.50 * X2))))
    a7 = 1 / (1 + math.exp(-(0.048 + (2.7 * a1) + (-0.85 * a2) + (0.79 * a3) + (-2.3 * a4) + (-1.9 * a5) + (0.60 * a6))))
    a8 = 1 / (1 + math.exp(-(-0.44 + (2.8 * a1) + (-3.1 * a2) + (6.1 * a3) + (-1.0 * a4) + (-1.3 * a5) + (0.098 * a6))))
    a9 = 1 / (1 + math.exp(-(0.39 + (-5.4 * a1) + (2.7 * a2) + (-0.51 * a3) + (3.2 * a4) + (2.5 * a5) + (-1.6 * a6))))
    a10 = 1 / (1 + math.exp(-(-0.22 + (1.7 * a1) + (-0.91 * a2) + (0.33 * a3) + (-1.7 * a4) + (-0.99 * a5) + (0.94 * a6))))
    a11 = 1 / (1 + math.exp(-(-1.9 + (3.4 * a7) + (0.71 * a8) + (-7.0 * a9) + (2.3 * a10))))
    a12 = 1 / (1 + math.exp(-(0.094 + (-0.98 * a7) + (-7.1 * a8) + (0.83 * a9) + (-0.95 * a10))))
    a13 = math.tanh(0.20 + (4.2 * a11) + (-4.8 * a12))
    return a13

# Crear un DataFrame con los datos de entrada
edad=int(input("Ingrese la edad: "))
colesterol=int(input("Ingrese el colesterol: "))
resultado = predecir_problema_cardiaco(edad, colesterol)
# Clasificación final
clase = 1 if resultado >= 0 else -1

print("Resultado de la predicción:", resultado)
print("Clase predicha:", clase)
if clase == 1:
    print("El paciente Si presenta problema cardíaco")
else:
    print("El paciente No presenta problema cardíaco")

