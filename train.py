import argparse
import pandas as pd
from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)
import os
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn


def parse_arguments():
    """
    Define los argumentos que se pueden modificar desde la terminal.
    """

    parser = argparse.ArgumentParser(
        description="Entrenamiento de Logistic Regression para detección de fallas AI4I."
    )

    parser.add_argument(
        "--C",
        type=float,
        default=1.0,
        help="Inverso de la regularización de Logistic Regression. Default: 1.0",
    )

    parser.add_argument(
        "--solver",
        type=str,
        default="lbfgs",
        choices=["lbfgs", "liblinear"],
        help="Algoritmo utilizado para optimizar Logistic Regression. Default: lbfgs",
    )

    parser.add_argument(
        "--class-weight",
        type=str,
        default="none",
        choices=["none", "balanced"],
        help="Pesos asignados a las clases. Default: none",
    )

    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Semilla aleatoria para reproducibilidad. Default: 42",
    )

    return parser.parse_args()


def load_data():
    """
    Descarga el dataset AI4I 2020 desde UCI y devuelve
    las variables predictoras X y la variable objetivo y.
    """

    dataset = fetch_ucirepo(id=601)

    X = dataset.data.features.copy()
    y = dataset.data.targets["Machine failure"].copy()

    return X, y


def prepare_data(X, y, random_state):
    """
    Divide los datos en entrenamiento y prueba y define
    el preprocesamiento para variables numéricas y categóricas.
    """

    numeric_features = [
        "Air temperature",
        "Process temperature",
        "Rotational speed",
        "Torque",
        "Tool wear",
    ]

    categorical_features = ["Type"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=random_state,
        stratify=y,
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_features),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore"),
                categorical_features,
            ),
        ]
    )

    return X_train, X_test, y_train, y_test, preprocessor


def train_model(
    X_train,
    y_train,
    preprocessor,
    C,
    solver,
    class_weight,
    random_state,
):
    """
    Construye y entrena el pipeline de preprocesamiento
    y Logistic Regression.
    """

    # argparse recibe "none" como texto.
    # Scikit-learn necesita None como valor de Python.
    if class_weight == "none":
        class_weight = None

    model = LogisticRegression(
        C=C,
        solver=solver,
        class_weight=class_weight,
        random_state=random_state,
        max_iter=1000,
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", model),
        ]
    )

    pipeline.fit(X_train, y_train)

    return pipeline


def evaluate_model(pipeline, X_test, y_test):
    """
    Evalúa el modelo con el conjunto de prueba y calcula
    las métricas principales y la matriz de confusión.
    """

    y_pred = pipeline.predict(X_test)

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
    }

    cm = confusion_matrix(y_test, y_pred)

    return metrics, cm


def save_confusion_matrix(cm):
    """
    Guarda la matriz de confusión como imagen para registrarla
    posteriormente como artefacto en MLflow.
    """

    output_path = "confusion_matrix.png"

    fig, ax = plt.subplots()

    image = ax.imshow(cm)

    ax.set_title("Matriz de confusión")
    ax.set_xlabel("Predicción")
    ax.set_ylabel("Valor real")

    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])

    for i in range(2):
        for j in range(2):
            ax.text(j, i, cm[i, j], ha="center", va="center")

    fig.colorbar(image)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

    return output_path


def main():
    args = parse_arguments()

    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("AI4I_Predictive_Maintenance")

    X, y = load_data()

    X_train, X_test, y_train, y_test, preprocessor = prepare_data(
        X,
        y,
        args.random_state,
    )

    pipeline = train_model(
        X_train,
        y_train,
        preprocessor,
        args.C,
        args.solver,
        args.class_weight,
        args.random_state,
    )

    metrics, cm = evaluate_model(
        pipeline,
        X_test,
        y_test,
    )

    artifact_path = save_confusion_matrix(cm)

    with mlflow.start_run(
        run_name=f"LR_C{args.C}_{args.solver}_{args.class_weight}"
    ):

        # Parámetros
        mlflow.log_param("C", args.C)
        mlflow.log_param("solver", args.solver)
        mlflow.log_param("class_weight", args.class_weight)
        mlflow.log_param("random_state", args.random_state)

        # Métricas
        mlflow.log_metric("accuracy", metrics["accuracy"])
        mlflow.log_metric("precision", metrics["precision"])
        mlflow.log_metric("recall", metrics["recall"])
        mlflow.log_metric("f1", metrics["f1"])

        # Artefacto
        mlflow.log_artifact(artifact_path)

        # Modelo completo
        mlflow.sklearn.log_model(
            pipeline,
            name="model",
        )



    print("Dataset cargado correctamente")
    print(f"Observaciones totales: {len(X)}")
    print(f"Datos de entrenamiento: {len(X_train)}")
    print(f"Datos de prueba: {len(X_test)}")
    print(f"Fallas totales: {y.sum()}")

    print()
    print("Configuración:")
    print(f"C = {args.C}")
    print(f"Solver = {args.solver}")
    print(f"Class weight = {args.class_weight}")
    print(f"Random state = {args.random_state}")

    print()
    print("Modelo entrenado correctamente")

    print()
    print("Resultados:")
    print(f"Accuracy:  {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall:    {metrics['recall']:.4f}")
    print(f"F1-score:  {metrics['f1']:.4f}")

    print()
    print("Matriz de confusión:")
    print(cm)


if __name__ == "__main__":
    main()