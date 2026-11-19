import math
import numpy as np

CONSEJOS = {
    "rodilla": "Flexiona más las rodillas durante la preparación del tiro.",
    "codo": "Busca una mayor extensión del codo al soltar el balón.",
    "hombro": "Revisa el ángulo del hombro al armar y liberar el lanzamiento.",
    "perfecto": "¡Excelente técnica de lanzamiento!",
}


def calcular_angulo_3pts(a, b, c) -> float:
    """Calcula el ángulo interior (en grados) en el punto medio b entre a y c."""
    a = np.array(a, dtype=float)
    b = np.array(b, dtype=float)
    c = np.array(c, dtype=float)

    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angulo = abs(radians * 180.0 / math.pi)
    return 360.0 - angulo if angulo > 180.0 else angulo


def evaluar_tecnica_tiro(
    min_rodilla: float | None,
    max_codo: float | None,
    angulo_hombro: float | None,
    velocidad_kmh: float = 0.0,
    tiro_id: int = 1,
) -> dict:
    """Evalúa los ángulos clave del tiro y devuelve puntaje, fallos y recomendaciones."""
    puntaje = 100.0
    fallos = []
    recomendaciones = []

    if min_rodilla is not None and min_rodilla > 130.0:
        puntaje -= 25.0
        fallos.append("Falta de flexión en las rodillas durante la preparación.")
        recomendaciones.append(CONSEJOS["rodilla"])

    if max_codo is not None and max_codo < 155.0:
        puntaje -= 25.0
        fallos.append("El codo no alcanza una extensión alta al soltar el balón.")
        recomendaciones.append(CONSEJOS["codo"])

    if angulo_hombro is not None and (angulo_hombro < 70.0 or angulo_hombro > 110.0):
        puntaje -= 15.0
        fallos.append("El ángulo del hombro al armar el tiro está fuera del rango de referencia.")
        recomendaciones.append(CONSEJOS["hombro"])

    puntaje = max(0.0, round(puntaje, 1))
    es_correcto = puntaje >= 80.0

    if not recomendaciones:
        recomendaciones.append(CONSEJOS["perfecto"])

    return {
        "id": tiro_id,
        "puntaje": puntaje,
        "correcto": es_correcto,
        "flexion_rodilla": round(min_rodilla, 1) if min_rodilla is not None else None,
        "extension_codo": round(max_codo, 1) if max_codo is not None else None,
        "angulo_hombro": round(angulo_hombro, 1) if angulo_hombro is not None else None,
        "velocidad_kmh": round(velocidad_kmh, 1),
        "fallos": fallos,
        "recomendaciones": recomendaciones,
    }