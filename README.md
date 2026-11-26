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

| Métrica | Resultado |
|---|---:|
| Recall de Fraude (1) | 71.43% |
| Precision de Fraude (1) | 0.71 |
| F1-score de Fraude (1) | 0.71 |
| Accuracy | 94% |
| ROC-AUC de prueba | 0.8319 |
| ROC-AUC de validación cruzada | 0.9263 |

### Mejores hiperparámetros encontrados

- criterion: entropy
- max_depth: 8
- n_estimators: 100

### Matriz de confusión

- 173 transacciones normales clasificadas correctamente.
- 6 transacciones normales clasificadas como fraude.
- 6 transacciones fraudulentas no detectadas.
- 15 transacciones fraudulentas detectadas correctamente.

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

Durante esta práctica aprendí a implementar y optimizar un modelo de Machine Learning utilizando Random Forest dentro de una instancia Ubuntu en AWS EC2.

De acuerdo con los resultados obtenidos, el modelo alcanzó un Recall del 71.43% en la clase Fraude (1), lo que significa que logró detectar correctamente 15 de las 21 transacciones fraudulentas. Sin embargo, quedaron 6 fraudes sin detectar, por lo que considero que todavía existe margen de mejora.
