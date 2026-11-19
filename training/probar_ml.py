import argparse
from pathlib import Path

import joblib
import pandas as pd

try:
    from training.extraer_angulos import COLUMNAS_CSV, extraer_angulos_video
except ModuleNotFoundError:
    from extraer_angulos import COLUMNAS_CSV, extraer_angulos_video

FEATURES_MODELO = [
    "min_rodilla",
    "max_codo",
    "angulo_hombro_max",
    "angulo_hombro_min",
]


def cargar_paquete_modelo(ruta_modelo: Path) -> dict:
    """Carga el paquete entrenado desde disco."""
    if not ruta_modelo.exists():
        raise FileNotFoundError(f"No se encontró el modelo entrenado: {ruta_modelo}")
    return joblib.load(ruta_modelo)


def predecir_desde_metricas(paquete: dict, metricas: dict) -> dict:
    """Ejecuta la predicción del modelo a partir de un diccionario de ángulos."""
    pipeline = paquete["pipeline"]
    features = paquete.get("features", FEATURES_MODELO)

    faltantes = [col for col in features if col not in metricas or metricas[col] is None]
    if faltantes:
        raise ValueError(f"Faltan métricas angulares para predecir: {faltantes}")

    fila_df = pd.DataFrame([{col: float(metricas[col]) for col in features}])
    prediccion = pipeline.predict(fila_df)[0]

    resultado = {
        "prediccion": str(prediccion),
        "objetivo": paquete.get("objetivo", "estilo_tiro"),
        "metricas": {col: float(metricas[col]) for col in features},
    }

    if hasattr(pipeline, "predict_proba"):
        probabilidades = pipeline.predict_proba(fila_df)[0]
        clases = list(pipeline.classes_)
        resultado["probabilidades"] = {
            str(clase): round(float(prob), 4)
            for clase, prob in zip(clases, probabilidades)
        }

    return resultado


def mostrar_comparacion_csv(ruta_csv: Path, metricas: dict):
    """Muestra cómo se ubican los ángulos frente a las filas guardadas en el CSV."""
    if not ruta_csv.exists() or ruta_csv.stat().st_size == 0:
        return

    df = pd.read_csv(ruta_csv)
    columnas_validas = [c for c in FEATURES_MODELO if c in df.columns]
    if df.empty or not columnas_validas:
        return

    print("\nComparación frente al promedio actual en el CSV:")
    for col in columnas_validas:
        promedio = float(df[col].mean())
        valor = float(metricas[col])
        diferencia = round(valor - promedio, 2)
        signo = "+" if diferencia >= 0 else ""
        print(f"  - {col}: actual={valor}° | promedio_csv={round(promedio, 2)}° ({signo}{diferencia}°)")


def main():
    parser = argparse.ArgumentParser(
        description="Prueba el modelo entrenado de tiro libre con un video o con ángulos manuales."
    )
    parser.add_argument(
        "--modelo",
        default="models/modelo_tiro_libre.joblib",
        help="Ruta al archivo .joblib entrenado",
    )
    parser.add_argument(
        "--video",
        help="Ruta a un video grabado de tiro libre para extraer ángulos y predecir",
    )
    parser.add_argument(
        "--rotacion",
        type=int,
        default=0,
        choices=[0, 90, 180, 270],
        help="Rotación horaria si el video viene girado",
    )
    parser.add_argument("--min-rodilla", type=float, help="Ángulo mínimo de rodilla (°)")
    parser.add_argument("--max-codo", type=float, help="Ángulo máximo de codo (°)")
    parser.add_argument("--hombro-max", type=float, help="Ángulo máximo de hombro (°)")
    parser.add_argument("--hombro-min", type=float, help="Ángulo mínimo de hombro (°)")
    parser.add_argument(
        "--csv",
        default="training/dataset_tiros.csv",
        help="CSV de referencia para comparar ángulos (opcional)",
    )
    args = parser.parse_args()

    paquete = cargar_paquete_modelo(Path(args.modelo))

    if args.video:
        metricas = extraer_angulos_video(args.video, rotacion=args.rotacion)
        if metricas is None:
            print("No se pudo detectar postura clara en el video.")
            return
    else:
        if None in (args.min_rodilla, args.max_codo, args.hombro_max, args.hombro_min):
            raise ValueError(
                "Pasa --video o bien los cuatro ángulos: --min-rodilla, --max-codo, --hombro-max y --hombro-min."
            )
        metricas = {
            "min_rodilla": args.min_rodilla,
            "max_codo": args.max_codo,
            "angulo_hombro_max": args.hombro_max,
            "angulo_hombro_min": args.hombro_min,
        }

    resultado = predecir_desde_metricas(paquete, metricas)
    print(f"Predicción ({resultado['objetivo']}): {resultado['prediccion']}")
    if "probabilidades" in resultado:
        print(f"Probabilidades / afinidad: {resultado['probabilidades']}")

    mostrar_comparacion_csv(Path(args.csv), metricas)


if __name__ == "__main__":
    main()