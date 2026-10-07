# Predicción de Fallas con MLflow

Proyecto de experimentación para detección de fallas utilizando el dataset
AI4I 2020 Predictive Maintenance de UCI.

El modelo utilizado es Logistic Regression y los experimentos son registrados
mediante MLflow Tracking.

## Instalación

Instalar las dependencias:

python -m pip install -r requirements.txt

## Ejecución

Ejecutar el modelo con los parámetros por defecto:

python train.py

También es posible especificar los hiperparámetros desde la línea de comandos:

python train.py --C 1.0 --solver lbfgs --class-weight balanced --random-state 42

Parámetros disponibles:

- `--C`: inverso de la regularización.
- `--solver`: algoritmo de optimización (`lbfgs` o `liblinear`).
- `--class-weight`: pesos de las clases (`none` o `balanced`).
- `--random-state`: semilla utilizada para reproducibilidad.

Los valores por defecto son `C=1.0`, `solver=lbfgs`,
`class_weight=none` y `random_state=42`.

## MLflow

Los experimentos se registran en:

`AI4I_Predictive_Maintenance`

Para iniciar la interfaz de MLflow en Windows:

python -m mlflow ui --backend-store-uri sqlite:///mlflow.db --workers 1

Después abrir en el navegador:

http://127.0.0.1:5000

Cada corrida registra los hiperparámetros, las métricas accuracy, precision,
recall y F1-score, la matriz de confusión como artefacto y el modelo entrenado.

## Reproducibilidad

Se utiliza `random_state=42` para mantener fija la partición de entrenamiento
y prueba. Al ejecutar nuevamente el script con los mismos argumentos se
obtienen los mismos resultados.