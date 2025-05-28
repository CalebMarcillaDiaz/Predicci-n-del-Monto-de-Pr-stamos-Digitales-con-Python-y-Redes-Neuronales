Predicción del Monto de Préstamos Digitales con Python y Redes Neuronales
Objetivo del trabajo
El objetivo de este proyecto es aplicar técnicas de regresión lineal y redes neuronales artificiales para predecir el monto de préstamos digitales otorgados a clientes peruanos, utilizando variables del historial bancario y socioeconómico de los clientes. El proyecto busca comparar el desempeño de ambos enfoques y documentar el proceso de análisis, modelado y visualización de resultados.

Breve descripción del dataset
El dataset utilizado, prestamos_digitales.csv, contiene información de clientes peruanos sobre transacciones digitales, características demográficas y montos de préstamos. Cada fila representa el resumen mensual de un cliente, e incluye variables como:

mes: Periodo de análisis (año/mes).
cliente: ID único del cliente.
rngSueldo: Rango salarial del cliente (variable categórica).
promSaldoBanco3Um: Promedio del saldo bancario en los últimos 3 meses.
ventaPrestDig: Monto de préstamo digital otorgado (variable objetivo).
Otras variables: edad, género, tipo de transacción digital, frecuencia, ubicación geográfica, etc.
El archivo utiliza punto y coma (“;”) como separador.

Librerías utilizadas
Python 3.8+
pandas
numpy
matplotlib
scikit-learn
tensorflow / keras
joblib (para guardar transformadores)
Explicación de los modelos
1. Regresión Lineal Simple
Se implementó un modelo de regresión lineal simple para predecir el monto del préstamo (ventaPrestDig) usando únicamente la variable promSaldoBanco3Um (promedio de saldo bancario en los últimos 3 meses). Se escogió esta variable porque refleja la capacidad financiera del cliente, lo que suele influir directamente en el monto de préstamo aprobado.

Procedimiento:

Limpieza de datos y eliminación de valores nulos.
División en sets de entrenamiento y prueba.
Ajuste del modelo y evaluación usando el error cuadrático medio (MSE).
Visualización de la relación entre el saldo promedio y el monto de préstamo.
2. Red Neuronal (Keras)
Se entrenó una red neuronal para predecir el monto de préstamo utilizando dos variables independientes:

rngSueldo (codificada con OneHotEncoder)
promSaldoBanco3Um (escalada entre 0 y 1)
Arquitectura:

Capa de entrada acorde al número de variables tras la codificación.
1 capa oculta (8 neuronas, activación ReLU).
1 capa oculta (4 neuronas, activación ReLU).
Capa de salida (1 neurona, salida continua).
Optimización con Adam, función de pérdida MSE y métrica MAE.
Entrenamiento:

100 épocas, batch size 16, 10% de validación.
Se guardó el modelo y los transformadores para uso futuro.
