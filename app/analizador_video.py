import math
import os
import shutil
import tempfile

import cv2
import mediapipe as mp
from moviepy import VideoFileClip

try:
    from app.biomecanica import calcular_angulo_3pts, evaluar_tecnica_tiro
except ModuleNotFoundError:
    from biomecanica import calcular_angulo_3pts, evaluar_tecnica_tiro

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils


def aplicar_clahe(frame):
    """Normaliza la iluminación para mejorar el contraste en canchas con sombras."""
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l_canal, a_canal, b_canal = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    l_mejorado = clahe.apply(l_canal)
    return cv2.cvtColor(cv2.merge((l_mejorado, a_canal, b_canal)), cv2.COLOR_LAB2BGR)


def _convertir_a_h264_moviepy(ruta_origen: str, ruta_destino: str, fps: float) -> bool:
    """Convierte el video generado por OpenCV a H.264 compatible con HTML5 usando MoviePy."""
    clip = None
    try:
        clip = VideoFileClip(ruta_origen)
        clip.write_videofile(
            ruta_destino,
            fps=fps,
            codec="libx264",
            audio=False,
            logger=None,
            ffmpeg_params=["-pix_fmt", "yuv420p", "-movflags", "+faststart"],
        )
        return os.path.exists(ruta_destino) and os.path.getsize(ruta_destino) > 0
    except Exception:
        return False
    finally:
        if clip is not None:
            try:
                clip.close()
            except Exception:
                pass


def analizar_video(
    ruta_video: str,
    ruta_video_salida: str | None = None,
    callback_progreso=None,
) -> dict:
    """Procesa un video grabado de entrenamiento individual enfocado en postura y ángulos."""
    if not os.path.exists(ruta_video):
        return {"error": f"No se encontró el archivo de video: {ruta_video}"}

    cap = cv2.VideoCapture(ruta_video)
    if not cap.isOpened():
        return {"error": "No se pudo abrir el video seleccionado."}

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
    ancho = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    alto = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    ruta_cv_raw = None
    writer = None
    if ruta_video_salida:
        temp_raw = tempfile.NamedTemporaryFile(delete=False, suffix="_raw.mp4")
        ruta_cv_raw = temp_raw.name
        temp_raw.close()
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(ruta_cv_raw, fourcc, fps, (ancho, alto))

    pose = mp_pose.Pose(
        static_image_mode=False,
        model_complexity=2,
        smooth_landmarks=True,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.6,
    )

    frame_count = 0
    cadera_previa = None
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

            if writer:
                writer.write(frame)
    finally:
        cap.release()
        if writer:
            writer.release()
        pose.close()

    ruta_final_video = None
    if ruta_video_salida and ruta_cv_raw and os.path.exists(ruta_cv_raw):
        if _convertir_a_h264_moviepy(ruta_cv_raw, ruta_video_salida, fps):
            ruta_final_video = ruta_video_salida
            try:
                os.remove(ruta_cv_raw)
            except OSError:
                pass
        else:
            shutil.move(ruta_cv_raw, ruta_video_salida)
            ruta_final_video = ruta_video_salida

    if not angulos_codo:
        return {"error": "No se pudo detectar la postura del jugador en el video."}

    min_rodilla = min(angulos_rodilla) if angulos_rodilla else None
    max_codo = max(angulos_codo) if angulos_codo else None
    angulo_hombro_ref = max(angulos_hombro) if angulos_hombro else None

    tiro = evaluar_tecnica_tiro(
        min_rodilla=min_rodilla,
        max_codo=max_codo,
        angulo_hombro=angulo_hombro_ref,
        velocidad_kmh=0.0,
        tiro_id=1,
    )

    tiro.setdefault("min_rodilla", min_rodilla)
    tiro.setdefault("max_codo", max_codo)
    tiro.setdefault("angulo_hombro", angulo_hombro_ref)

    return {
        "modo_analisis": "postura_mediapipe",
        "ruta_video_salida": ruta_final_video,
        "resumen": {
            "tiros_detectados": 1,
            "tiros_correctos": 1 if tiro.get("correcto") else 0,
            "puntaje_promedio": tiro.get("puntaje", 0),
            "velocidad_maxima": 0.0,
        },
        "tiros": [tiro],
        "fallos_frecuentes": tiro.get("fallos", []),
        "recomendaciones": tiro.get("recomendaciones", []),
    }