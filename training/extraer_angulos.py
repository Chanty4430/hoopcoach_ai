import argparse
import csv
import math
import os
from pathlib import Path

import cv2
import mediapipe as mp

mp_pose = mp.solutions.pose

COLUMNAS_CSV = [
    "archivo_video",
    "jugador_o_etiqueta",
    "estilo_tiro",
    "min_rodilla",
    "max_codo",
    "angulo_hombro_max",
    "angulo_hombro_min",
]


def calcular_angulo_3pts(a, b, c) -> float:
    """Calcula el ángulo en grados formado por tres puntos (b es el vértice)."""
    ab = (a[0] - b[0], a[1] - b[1])
    cb = (c[0] - b[0], c[1] - b[1])
    producto_punto = ab[0] * cb[0] + ab[1] * cb[1]
    norma_ab = math.hypot(ab[0], ab[1])
    norma_cb = math.hypot(cb[0], cb[1])
    if norma_ab == 0 or norma_cb == 0:
        return 0.0
    cos_angulo = max(-1.0, min(1.0, producto_punto / (norma_ab * norma_cb)))
    return round(math.degrees(math.acos(cos_angulo)), 2)


def rotar_frame(frame, rotacion: int):
    """Rota un frame según 0, 90, 180 o 270 grados en sentido horario."""
    if rotacion == 90:
        return cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    if rotacion == 180:
        return cv2.rotate(frame, cv2.ROTATE_180)
    if rotacion == 270:
        return cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return frame


def extraer_angulos_video(ruta_video: str, rotacion: int = 0) -> dict | None:
    """Procesa un video grabado de tiro libre y extrae métricas angulares clave."""
    if not os.path.exists(ruta_video):
        raise FileNotFoundError(f"No existe el video: {ruta_video}")

    cap = cv2.VideoCapture(ruta_video)
    if not cap.isOpened():
        raise RuntimeError(f"No se pudo abrir el video: {ruta_video}")

    angulos_rodilla = []
    angulos_codo = []
    angulos_hombro = []

    pose = mp_pose.Pose(
        static_image_mode=False,
        model_complexity=2,
        smooth_landmarks=True,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.6,
    )

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret or frame is None:
                break

            frame_rotado = rotar_frame(frame, rotacion)
            alto, ancho = frame_rotado.shape[:2]
            frame_rgb = cv2.cvtColor(frame_rotado, cv2.COLOR_BGR2RGB)
            resultados = pose.process(frame_rgb)

            if resultados.pose_landmarks:
                lm = resultados.pose_landmarks.landmark

                cadera = [
                    lm[mp_pose.PoseLandmark.RIGHT_HIP.value].x * ancho,
                    lm[mp_pose.PoseLandmark.RIGHT_HIP.value].y * alto,
                ]
                rodilla = [
                    lm[mp_pose.PoseLandmark.RIGHT_KNEE.value].x * ancho,
                    lm[mp_pose.PoseLandmark.RIGHT_KNEE.value].y * alto,
                ]
                tobillo = [
                    lm[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x * ancho,
                    lm[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y * alto,
                ]
                hombro = [
                    lm[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x * ancho,
                    lm[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y * alto,
                ]
                codo = [
                    lm[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x * ancho,
                    lm[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y * alto,
                ]
                muneca = [
                    lm[mp_pose.PoseLandmark.RIGHT_WRIST.value].x * ancho,
                    lm[mp_pose.PoseLandmark.RIGHT_WRIST.value].y * alto,
                ]

                angulos_rodilla.append(calcular_angulo_3pts(cadera, rodilla, tobillo))
                angulos_codo.append(calcular_angulo_3pts(hombro, codo, muneca))
                angulos_hombro.append(calcular_angulo_3pts(cadera, hombro, codo))
    finally:
        cap.release()
        pose.close()

    if not angulos_codo or not angulos_rodilla or not angulos_hombro:
        return None

    return {
        "min_rodilla": min(angulos_rodilla),
        "max_codo": max(angulos_codo),
        "angulo_hombro_max": max(angulos_hombro),
        "angulo_hombro_min": min(angulos_hombro),
    }


def guardar_en_csv(ruta_csv: Path, fila: dict):
    """Guarda o anexa una fila con los ángulos extraídos en el CSV."""
    ruta_csv.parent.mkdir(parents=True, exist_ok=True)
    archivo_existe = ruta_csv.exists() and ruta_csv.stat().st_size > 0

    with open(ruta_csv, mode="a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNAS_CSV)
        if not archivo_existe:
            writer.writeheader()
        writer.writerow(fila)


def main():
    parser = argparse.ArgumentParser(description="Extrae ángulos de tiro libre a un CSV.")
    parser.add_argument("--video", required=True, help="Ruta al video grabado de tiro libre")
    parser.add_argument("--etiqueta", required=True, help="Ej: santiago_tiro1, stephen_curry, steve_nash")
    parser.add_argument(
        "--estilo",
        default="tradicional",
        help="Ej: tradicional (Curry, Nash, Price, Lillard) o cuchara (Rick Barry)",
    )
    parser.add_argument(
        "--rotacion",
        type=int,
        default=0,
        choices=[0, 90, 180, 270],
        help="Rotación horaria para corregir videos girados",
    )
    parser.add_argument(
        "--csv",
        default="training/datasets/dataset_tiros.csv",
        help="Ruta del archivo CSV donde se guardarán las muestras",
    )
    args = parser.parse_args()

    metricas = extraer_angulos_video(args.video, rotacion=args.rotacion)
    if metricas is None:
        print("No se detectó postura clara en el video.")
        return

    fila = {
        "archivo_video": os.path.basename(args.video),
        "jugador_o_etiqueta": args.etiqueta,
        "estilo_tiro": args.estilo,
        **metricas,
    }
    ruta_csv = Path(args.csv)
    guardar_en_csv(ruta_csv, fila)
    print(f"Guardado en {ruta_csv}: {fila}")


if __name__ == "__main__":
    main()