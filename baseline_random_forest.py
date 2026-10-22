import os
import time
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    roc_auc_score, f1_score, precision_score, recall_score,
    confusion_matrix, classification_report
)
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

# Horizonte de prediccion a usar: "1year" (predice quiebra a 5 anios vista)
# hasta "5year" (predice quiebra al anio siguiente). Empezamos con "1year"
# porque es el mas grande (7027 filas) y el mas usado en la literatura.
YEAR = "1year"


def cargar_datos(year=YEAR):
    """Carga el dataset, ya sea via ucimlrepo (internet) o desde un .arff local."""

    # --- Intento 1: libreria oficial de UCI (necesita internet) ---
    try:
        from ucimlrepo import fetch_ucirepo
        print("Intentando descargar el dataset via ucimlrepo...")
        dataset = fetch_ucirepo(id=365)
        X = dataset.data.features
        y = dataset.data.targets
        # ucimlrepo entrega los 5 anios juntos; si tu version los separa,
        # ajusta este bloque segun la estructura que te devuelva.
        df = pd.concat([X, y], axis=1)
        df.columns = [f"Attr{i+1}" for i in range(X.shape[1])] + ["class"]
        print(f"Dataset cargado via ucimlrepo: {df.shape[0]} filas, {df.shape[1]} columnas.")
        return df
    except Exception as e:
        print(f"No se pudo usar ucimlrepo ({e}). Buscando archivo .arff local...")

    # --- Intento 2: archivo .arff local descargado manualmente ---
    # arff_path = os.path.join(DATA_DIR, f"{year}.arff")
    # if os.path.exists(arff_path):
    #     from scipy.io import arff
    #     data, meta = arff.loadarff(arff_path)
    #     df = pd.DataFrame(data)
    #     # La columna 'class' en el .arff viene como bytes (b'0'/b'1'); se convierte a int
    #     df["class"] = df["class"].apply(lambda v: int(v.decode("utf-8")) if isinstance(v, bytes) else int(v))
    #     print(f"Dataset cargado desde {arff_path}: {df.shape[0]} filas, {df.shape[1]} columnas.")
    #     return df

    raise FileNotFoundError(
        f"No se encontro el dataset. Instala 'ucimlrepo' (pip install ucimlrepo) "
        f"con internet disponible, o coloca manualmente '{year}.arff' en la carpeta data/. "
        f"Ver instrucciones al inicio de este archivo."
    )


def preparar_datos(df):
    """Separa features/target y hace una imputacion simple de valores faltantes.

    NOTA: esta imputacion (mediana) es solo para tener una linea base valida.
    Mas adelante, la version DIFUSA del algoritmo debe manejar estos valores
    faltantes/imprecisos de forma nativa (funciones de membresia), en vez de
    imputarlos de forma "cruda" como aqui.
    """
    y = df["class"].astype(int)
    X = df.drop(columns=["class"])

    print(f"\nDistribucion de clases (0 = no quiebra, 1 = quiebra):")
    print(y.value_counts())
    print(f"Porcentaje de quiebra: {100 * y.mean():.2f}%")

    print(f"\nValores faltantes totales: {X.isna().sum().sum()} de {X.size} celdas "
          f"({100 * X.isna().sum().sum() / X.size:.2f}%)")

    imputer = SimpleImputer(strategy="median")
    X_imputado = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

    return X_imputado, y


def entrenar_y_evaluar(X, y, n_estimators=200, random_state=42):
    """Entrena un Random Forest clasico y mide precision + tiempo de computo."""

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, stratify=y, random_state=random_state
    )

    modelo = RandomForestClassifier(
        n_estimators=n_estimators,
        class_weight="balanced",  # compensa el desbalance de clases (pocas quiebras)
        random_state=random_state,
        n_jobs=-1,
    )

    # --- Medicion de tiempo de entrenamiento (linea base para el analisis de complejidad) ---
    t0 = time.perf_counter()
    modelo.fit(X_train, y_train)
    tiempo_entrenamiento = time.perf_counter() - t0

    # --- Medicion de tiempo de prediccion ---
    t0 = time.perf_counter()
    y_pred = modelo.predict(X_test)
    y_proba = modelo.predict_proba(X_test)[:, 1]
    tiempo_prediccion = time.perf_counter() - t0

    metricas = {
        "n_train": len(X_train),
        "n_test": len(X_test),
        "n_estimators": n_estimators,
        "tiempo_entrenamiento_s": tiempo_entrenamiento,
        "tiempo_prediccion_s": tiempo_prediccion,
        "auc_roc": roc_auc_score(y_test, y_proba),
        "f1_score": f1_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
    }

    print("\n" + "=" * 60)
    print("RESULTADOS - Random Forest clasico (linea base)")
    print("=" * 60)
    for k, v in metricas.items():
        print(f"  {k:25s}: {v}")

    print("\nMatriz de confusion:")
    print(confusion_matrix(y_test, y_pred))
    print("\nReporte de clasificacion:")
    print(classification_report(y_test, y_pred, target_names=["No quiebra", "Quiebra"]))

    return modelo, metricas


def analisis_escalabilidad(X, y, tamanos=(0.2, 0.4, 0.6, 0.8, 1.0), n_estimators=200):
    """Mide como crece el tiempo de entrenamiento al variar el tamano del dataset.

    Esto es el punto de partida empirico para el objetivo especifico de
    'medir empiricamente el tiempo al variar el tamano del dataset', que
    luego se comparara contra la version difusa + genetica del algoritmo.
    """
    resultados = []
    n_total = len(X)

    for frac in tamanos:
        n = int(n_total * frac)
        X_sub, y_sub = X.iloc[:n], y.iloc[:n]

        t0 = time.perf_counter()
        modelo = RandomForestClassifier(
            n_estimators=n_estimators, class_weight="balanced",
            random_state=42, n_jobs=-1,
        )
        modelo.fit(X_sub, y_sub)
        tiempo = time.perf_counter() - t0

        resultados.append({"fraccion": frac, "n_filas": n, "tiempo_entrenamiento_s": tiempo})
        print(f"  Fraccion {frac:.1f} ({n:5d} filas): {tiempo:.4f} s")

    return pd.DataFrame(resultados)


def main():
    print("Cargando dataset...")
    df = cargar_datos()

    print("\nPreparando datos (separacion X/y e imputacion de valores faltantes)...")
    X, y = preparar_datos(df)

    print("\nEntrenando y evaluando Random Forest clasico (linea base)...")
    modelo, metricas = entrenar_y_evaluar(X, y)

    pd.DataFrame([metricas]).to_csv(
        os.path.join(RESULTS_DIR, "baseline_metricas.csv"), index=False
    )

    print("\nAnalisis de escalabilidad (tiempo de entrenamiento vs. tamano del dataset)...")
    df_escalabilidad = analisis_escalabilidad(X, y)
    df_escalabilidad.to_csv(
        os.path.join(RESULTS_DIR, "baseline_escalabilidad.csv"), index=False
    )

    print(f"\nResultados guardados en: {RESULTS_DIR}")
    print("Listo. Este es el punto de comparacion para el algoritmo hibrido "
          "(Random Forest difuso + Algoritmo Genetico) que se implementara despues.")


if __name__ == "__main__":
    main()  