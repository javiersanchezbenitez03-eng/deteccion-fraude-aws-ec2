# Detección de fraude bancario en AWS EC2

## Descripción del proyecto

Esta práctica consiste en implementar un modelo de Machine Learning para detectar transacciones bancarias fraudulentas utilizando Random Forest en una instancia Ubuntu de AWS EC2.

## Objetivo

Aplicar técnicas de preparación de datos, validación cruzada, optimización de hiperparámetros, evaluación de métricas y serialización de modelos.

## Tecnologías utilizadas

- AWS Academy Learner Lab
- AWS EC2
- Ubuntu
- Python
- Pandas
- NumPy
- Scikit-learn
- Joblib
- Boto3
- GitHub

## Metodología

1. Generación de 1,000 transacciones bancarias simuladas.
2. División de datos en 80% de entrenamiento y 20% de prueba.
3. Escalado de características mediante StandardScaler.
4. Entrenamiento mediante Random Forest.
5. Optimización de hiperparámetros con GridSearchCV.
6. Validación cruzada de cinco particiones.
7. Evaluación de métricas de clasificación.
8. Serialización del modelo y escalador.

## Archivos del proyecto

| Archivo | Descripción |
|---|---|
| practica_fraude_ec2.py | Código principal del modelo. |
| modelo_fraude_opt.pkl | Modelo entrenado y optimizado. |
| scaler_fraude.pkl | Escalador de características. |
| resultados.txt | Resultados del entrenamiento. |
| requirements.txt | Dependencias del proyecto. |

## Resultados obtenidos

- Recall de Fraude (1): [valor]
- ROC-AUC Score: [valor]
- Mejores hiperparámetros: [valores]

## Instalación y ejecución

Crear un entorno virtual:

`python3 -m venv venv`

Activar el entorno:

`source venv/bin/activate`

Instalar dependencias:

`pip install -r requirements.txt`

Ejecutar el programa:

`python3 practica_fraude_ec2.py`

## Conclusión personal

Durante esta práctica aprendí a implementar y optimizar un modelo de Machine Learning en AWS EC2.

El modelo obtuvo un Recall de [valor], logrando detectar el [porcentaje]% de los fraudes reales, y un ROC-AUC Score de [valor], que permite evaluar su capacidad de clasificación.

La serialización del modelo y del escalador facilita su reutilización en futuros proyectos, permitiendo almacenarlos en AWS S3 o integrarlos en una API de predicción sin necesidad de repetir el entrenamiento.
