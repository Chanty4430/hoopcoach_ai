import os
from pathlib import Path
import tempfile

import plotly.graph_objects as go
import streamlit as st

try:
    from app.analizador_video import analizar_video
except ModuleNotFoundError:
    from analizador_video import analizar_video


st.set_page_config(
    page_title="HoopCoach AI",
    page_icon="🏀",
    layout="wide",
)

st.title("🏀 HoopCoach AI")
st.caption("Análisis biomecánico de postura en videos grabados de lanzamiento en baloncesto.")

if "resultado_analisis" not in st.session_state:
    st.session_state.resultado_analisis = None
if "video_bytes_salida" not in st.session_state:
    st.session_state.video_bytes_salida = None

st.sidebar.header("Cargar video de entrenamiento")
archivo_video = st.sidebar.file_uploader(
    "Sube un video de tu sesión",
    type=["mp4", "mov", "avi"],
)

if archivo_video is not None:
    st.sidebar.success(f"Video listo: {archivo_video.name}")
    analizar_btn = st.sidebar.button("🚀 Analizar postura", type="primary")
else:
    analizar_btn = False
    st.info("Sube un video grabado en la barra lateral para comenzar el análisis de postura.")

if archivo_video is not None and analizar_btn:
    progreso_bar = st.progress(0, text="Iniciando análisis...")

    def actualizar_progreso(valor: float):
        progreso_bar.progress(
            min(max(valor, 0.0), 1.0),
            text=f"Analizando postura... {int(min(max(valor, 0.0), 1.0) * 100)}%",
        )

    suffix = Path(archivo_video.name).suffix or ".mp4"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_in:
        temp_in.write(archivo_video.getbuffer())
        ruta_entrada = temp_in.name

    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_out:
        ruta_salida = temp_out.name

    try:
        datos = analizar_video(
            ruta_video=ruta_entrada,
            ruta_video_salida=ruta_salida,
            callback_progreso=actualizar_progreso,
        )
        progreso_bar.progress(1.0, text="Análisis completado.")

        if "error" not in datos and os.path.exists(ruta_salida) and os.path.getsize(ruta_salida) > 0:
            with open(ruta_salida, "rb") as f_out:
                st.session_state.video_bytes_salida = f_out.read()
        else:
            st.session_state.video_bytes_salida = None

        st.session_state.resultado_analisis = datos
    finally:
        if os.path.exists(ruta_entrada):
            os.remove(ruta_entrada)
        if os.path.exists(ruta_salida):
            try:
                os.remove(ruta_salida)
            except OSError:
                pass

datos = st.session_state.resultado_analisis
video_bytes = st.session_state.video_bytes_salida

if datos is not None:
    if "error" in datos:
        st.error(datos["error"])
    else:
        st.caption("Modo actual: análisis biomecánico de postura con MediaPipe")

        resumen = datos.get("resumen", {})
        col1, col2, col3 = st.columns(3)
        col1.metric("Tiros evaluados", resumen.get("tiros_detectados", 1))
        col2.metric("Posturas correctas", resumen.get("tiros_correctos", 0))
        col3.metric("Puntaje promedio", f"{resumen.get('puntaje_promedio', 0)} / 100")

        tiros = datos.get("tiros", [])
        if tiros:
            tiro_actual = tiros[0]
            st.subheader("Ángulos articulares registrados")

            ang_col1, ang_col2, ang_col3 = st.columns(3)
            ang_col1.metric(
                "Flexión mínima rodilla",
                f"{tiro_actual.get('min_rodilla', 0):.1f}°" if tiro_actual.get("min_rodilla") is not None else "N/A",
            )
            ang_col2.metric(
                "Extensión máxima codo",
                f"{tiro_actual.get('max_codo', 0):.1f}°" if tiro_actual.get("max_codo") is not None else "N/A",
            )
            ang_col3.metric(
                "Ángulo de hombro",
                f"{tiro_actual.get('angulo_hombro', 0):.1f}°" if tiro_actual.get("angulo_hombro") is not None else "N/A",
            )

            fig = go.Figure(
                data=[
                    go.Bar(
                        x=["Rodilla (mín)", "Codo (máx)", "Hombro"],
                        y=[
                            tiro_actual.get("min_rodilla") or 0,
                            tiro_actual.get("max_codo") or 0,
                            tiro_actual.get("angulo_hombro") or 0,
                        ],
                    )
                ]
            )
            fig.update_layout(
                title="Resumen de ángulos articulares (°)",
                yaxis_title="Grados (°)",
                height=340,
            )
            st.plotly_chart(fig, width="stretch")

        fallos = datos.get("fallos_frecuentes", [])
        recomendaciones = datos.get("recomendaciones", [])

        rec_col1, rec_col2 = st.columns(2)
        with rec_col1:
            st.subheader("Aspectos a corregir")
            if fallos:
                for fallo in fallos:
                    st.write(f"- {fallo}")
            else:
                st.write("No se detectaron fallos relevantes en la postura.")

        with rec_col2:
            st.subheader("Recomendaciones")
            if recomendaciones:
                for rec in recomendaciones:
                    st.write(f"- {rec}")
            else:
                st.write("Mantén la consistencia de la mecánica registrada.")

        if video_bytes:
            st.subheader("Video procesado con esqueleto de postura")
            st.video(video_bytes, format="video/mp4")