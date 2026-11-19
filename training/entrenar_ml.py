import argparse
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

COLUMNAS_FEATURES = [
    "min_rodilla",
    "max_codo",
    "angulo_hombro_max",
    "angulo_hombro_min",
]


def entrenar_modelo(
    ruta_csv: Path,
    ruta_modelo: Path,
    objetivo: str = "estilo_tiro",
    n_estimadores: int = 100,
) -> dict:
    """Entrena un clasificador tabular de tiro libre a partir del CSV de ángulos."""
    if not ruta_csv.exists():
        raise FileNotFoundError(f"No se encontró el dataset CSV: {ruta_csv}")

    df = pd.read_csv(ruta_csv)
    if df.empty:
        raise ValueError("El archivo CSV está vacío. Extrae al menos algunos tiros primero.")

    faltantes = [col for col in COLUMNAS_FEATURES + [objetivo] if col not in df.columns]
    if faltantes:
        raise ValueError(f"Faltan columnas requeridas en el CSV: {faltantes}")

    df_limpio = df.dropna(subset=COLUMNAS_FEATURES + [objetivo]).copy()
    if df_limpio.empty:
        raise ValueError("No hay filas válidas sin valores nulos en las columnas de entrenamiento.")

    X = df_limpio[COLUMNAS_FEATURES].astype(float)
    y = df_limpio[objetivo].astype(str)

    clases_unicas = y.nunique()
    total_muestras = len(df_limpio)

    pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "clf",
                RandomForestClassifier(
                    n_estimators=n_estimadores,
                    random_state=42,
                ),
            ),
        ]
    )

    metricas = {"muestras_totales": total_muestras, "clases": sorted(y.unique().tolist())}

    # Con muy pocas muestras o 1 sola clase, ajustamos en todo el dataset para validar el pipeline
    if total_muestras < 6 or clases_unicas < 2:
        pipeline.fit(X, y)
        metricas["modo_entrenamiento"] = "ajuste_base_pocas_muestras"
        metricas["nota"] = (
            f"Se entrenó con {total_muestras} muestra(s) y {clases_unicas} clase(s) en '{objetivo}'. "
            "Suma más tiros (y distintas etiquetas/jugadores de referencia como Curry, Nash, Price, Barry o Lillard) "
            "para evaluar con división train/test real."
        )
    else:
        test_size = 0.25 if total_muestras >= 8 else 0.2
        stratify_param = y if y.value_counts().min() >= 2 else None
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=test_size,
            random_state=42,
            stratify=stratify_param,
        )
        pipeline.fit(X_train, y_train)
        predicciones = pipeline.predict(X_test)
        metricas["modo_entrenamiento"] = "train_test_split"
        metricas["accuracy"] = round(float(accuracy_score(y_test, predicciones)), 4)
        metricas["reporte"] = classification_report(y_test, predicciones, zero_division=0)

    ruta_modelo.parent.mkdir(parents=True, exist_ok=True)
    paquete = {
        "pipeline": pipeline,
        "features": COLUMNAS_FEATURES,
        "objetivo": objetivo,
        "clases": sorted(y.unique().tolist()),
    }
    joblib.dump(paquete, ruta_modelo)
    metricas["ruta_modelo"] = str(ruta_modelo)
    return metricas


def main():
    parser = argparse.ArgumentParser(
        description="Entrena el modelo de Machine Learning de tiro libre a partir de ángulos."
    )
    parser.add_argument(
        "--csv",
        default="training/dataset_tiros.csv",
        help="Ruta al CSV generado por extraer_angulos.py",
    )
    parser.add_argument(
        "--salida",
        default="models/modelo_tiro_libre.joblib",
        help="Ruta donde se guardará el modelo entrenado",
    )
    parser.add_argument(
        "--objetivo",
        default="estilo_tiro",
        help="Columna objetivo a predecir (ej: estilo_tiro o jugador_o_etiqueta)",
    )
    parser.add_argument(
        "--arboles",
        type=int,
        default=100,
        help="Número de árboles para RandomForestClassifier",
    )
    args = parser.parse_args()

    resultado = entrenar_modelo(
        ruta_csv=Path(args.csv),
        ruta_modelo=Path(args.salida),
        objetivo=args.objetivo,
        n_estimadores=args.arboles,
    )

    print("Entrenamiento completado:")
    for clave, valor in resultado.items():
        if clave == "reporte":
            print("\nReporte de clasificación:\n" + valor)
        else:
            print(f"  - {clave}: {valor}")


if __name__ == "__main__":
    main()