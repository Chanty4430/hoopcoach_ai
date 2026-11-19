import math
import os
from pathlib import Path

import cv2
import mediapipe as mp
from ultralytics import YOLO

try:
    from app.biomecanica import calcular_angulo_3pts, evaluar_tecnica_tiro
except ModuleNotFoundError:
    from biomecanica import calcular_angulo_3pts, evaluar_tecnica_tiro

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"


def _resolver_ruta_modelo() -> str:
    """Busca un modelo entrenado en models/best.pt; si no existe, usa yolo11n.pt."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    best_model = MODELS_DIR / "best.pt"
    base_model = MODELS_DIR / "yolo11n.pt"

    if best_model.exists():
        return str(best_model)
    return str(base_model)


def aplicar_clahe(frame):
    """Normaliza la iluminación para mejorar el contraste en canchas con sombras."""
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l_canal, a_canal, b_canal = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    l_mejorado = clahe.apply(l_canal)
    return cv2.cvtColor(cv2.merge((l_mejorado, a_canal, b_canal)), cv2.COLOR_LAB2BGR)


def analizar_video(ruta_video: str, ruta_video_salida: str | None = None, callback_progreso=None) -> dict:
    """Procesa un video grabado de entrenamiento individual y devuelve métricas biomecánicas."""
    if not os.path.exists(ruta_video):
        return {"error": f"No se encontró el archivo de video: {ruta_video}"}

    cap = cv2.VideoCapture(ruta_video)
    if not cap.isOpened():
        return {"error": "No se pudo abrir el video seleccionado."}

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
    ancho = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    alto = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Estimación sencilla de escala píxel-a-metro basada en alto del cuadro
    factor_px_a_m = 2.0 / max(alto, 1)
    tiempo_frame_s = 1.0 / max(fps, 1.0)

    writer = None
    if ruta_video_salida:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(ruta_video_salida, fourcc, fps, (ancho, alto))

    ruta_modelo = _resolver_ruta_modelo()
    modelo = YOLO(ruta_modelo)
    es_modelo_propio = Path(ruta_modelo).name == "best.pt"
    clases_balon = [0] if es_modelo_propio else [32]

    pose = mp_pose.Pose(
        static_image_mode=False,
        model_complexity=2,
        smooth_landmarks=True,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.6,
    )

    frame_count = 0
    cadera_previa = None
    posicion_balon_previa = None
    historial_velocidades = []
    angulos_rodilla = []
    angulos_codo = []
    angulos_hombro = []

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret or frame is None:
                break

            frame_count += 1
            if callback_progreso and frame_count % 5 == 0:
                callback_progreso(min(frame_count / total_frames, 0.95))

            frame_procesado = aplicar_clahe(frame)
            frame_rgb = cv2.cvtColor(frame_procesado, cv2.COLOR_BGR2RGB)

            # 1. Estimación de pose con MediaPipe
            resultados_pose = pose.process(frame_rgb)
            if resultados_pose.pose_landmarks:
                lm = resultados_pose.pose_landmarks.landmark
                cadera_x = lm[mp_pose.PoseLandmark.RIGHT_HIP.value].x * ancho
                cadera_y = lm[mp_pose.PoseLandmark.RIGHT_HIP.value].y * alto

                es_mismo_jugador = True
                if cadera_previa is not None:
                    salto_px = math.hypot(cadera_x - cadera_previa[0], cadera_y - cadera_previa[1])
                    if salto_px > 180:
                        es_mismo_jugador = False

                if es_mismo_jugador:
                    cadera_previa = (cadera_x, cadera_y)

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
                    cadera = [cadera_x, cadera_y]
                    rodilla = [
                        lm[mp_pose.PoseLandmark.RIGHT_KNEE.value].x * ancho,
                        lm[mp_pose.PoseLandmark.RIGHT_KNEE.value].y * alto,
                    ]
                    tobillo = [
                        lm[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x * ancho,
                        lm[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y * alto,
                    ]

                    angulos_rodilla.append(calcular_angulo_3pts(cadera, rodilla, tobillo))
                    angulos_codo.append(calcular_angulo_3pts(hombro, codo, muneca))
                    angulos_hombro.append(calcular_angulo_3pts(cadera, hombro, codo))

                    mp_drawing.draw_landmarks(
                        frame,
                        resultados_pose.pose_landmarks,
                        mp_pose.POSE_CONNECTIONS,
                        mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=3),
                        mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2),
                    )

            # 2. Detección de balón con YOLO
            resultados_yolo = modelo.predict(
                frame_procesado,
                conf=0.12,
                classes=clases_balon,
                verbose=False,
            )

            vel_actual_kmh = 0.0
            if resultados_yolo and len(resultados_yolo[0].boxes) > 0:
                caja = resultados_yolo[0].boxes[0]
                x1, y1, x2, y2 = caja.xyxy[0].cpu().numpy()
                cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)

                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 165, 255), 2)
                cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)

                if posicion_balon_previa is not None:
                    dist_px = math.hypot(cx - posicion_balon_previa[0], cy - posicion_balon_previa[1])
                    dist_m = dist_px * factor_px_a_m
                    vel_actual_kmh = (dist_m / tiempo_frame_s) * 3.6
                    if vel_actual_kmh < 120.0:
                        historial_velocidades.append(vel_actual_kmh)

                posicion_balon_previa = (cx, cy)
            else:
                posicion_balon_previa = None

            cv2.putText(
                frame,
                f"Velocidad: {vel_actual_kmh:.1f} km/h",
                (25, 45),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.85,
                (0, 255, 255),
                2,
            )

            if writer:
                writer.write(frame)
    finally:
        cap.release()
        if writer:
            writer.release()
        pose.close()

    if not angulos_codo:
        return {"error": "No se pudo detectar la postura del jugador en el video."}

    min_rodilla = min(angulos_rodilla) if angulos_rodilla else None
    max_codo = max(angulos_codo) if angulos_codo else None
    angulo_hombro_ref = max(angulos_hombro) if angulos_hombro else None
    vel_maxima = max(historial_velocidades) if historial_velocidades else 0.0

    tiro = evaluar_tecnica_tiro(
        min_rodilla=min_rodilla,
        max_codo=max_codo,
        angulo_hombro=angulo_hombro_ref,
        velocidad_kmh=vel_maxima,
        tiro_id=1,
    )

    return {
        "modelo_usado": Path(ruta_modelo).name,
        "resumen": {
            "tiros_detectados": 1,
            "tiros_correctos": 1 if tiro["correcto"] else 0,
            "puntaje_promedio": tiro["puntaje"],
            "velocidad_maxima": tiro["velocidad_kmh"],
        },
        "tiros": [tiro],
        "fallos_frecuentes": tiro["fallos"],
        "recomendaciones": tiro["recomendaciones"],
        "velocidades": [round(v, 1) for v in historial_velocidades],
    }