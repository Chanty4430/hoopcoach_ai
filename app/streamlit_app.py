import os
import tempfile
from pathlib import Path

import plotly.graph_objects as go
import streamlit as st
from moviepy import VideoFileClip

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
st.markdown("Análisis biomecánico de videos grabados de lanzamiento en baloncesto.")

# Barra lateral: carga del video grabado
st.sidebar.header("Cargar video de entrenamiento")
archivo_video = st.sidebar.file_uploader(
    "Sube un video de tu sesión",
    type=["mp4", "mov", "avi"],
)

if archivo_video is not None:
    sufijo_entrada = Path(archivo_video.name).suffix or ".mp4"

    with tempfile.NamedTemporaryFile(delete=False, suffix=sufijo_entrada) as tmp_in:
        tmp_in.write(archivo_video.read())
        ruta_tmp_in = tmp_in.name

    ruta_tmp_out = tempfile.mktemp(suffix="_procesado.mp4")
    ruta_tmp_web = tempfile.mktemp(suffix="_web.mp4")

    st.sidebar.success(f"Video listo: {archivo_video.name}")

    if st.sidebar.button("🚀 Analizar lanzamiento", type="primary"):
        barra = st.progress(0.0, text="Iniciando análisis...")
        datos = None

        def actualizar_progreso(valor: float):
            barra.progress(min(max(valor, 0.0), 0.95), text=f"Procesando video... {int(valor * 100)}%")

        try:
            with st.spinner("Analizando postura y balón frame a frame..."):
                datos = analizar_video(
                    ruta_video=ruta_tmp_in,
                    ruta_video_salida=ruta_tmp_out,
                    callback_progreso=actualizar_progreso,
                )
        finally:
            if os.path.exists(ruta_tmp_in):
                try:
                    os.unlink(ruta_tmp_in)
                except OSError:
                    pass

        barra.progress(1.0, text="¡Análisis completado!")

        if not datos or "error" in datos:
            mensaje_error = datos.get("error", "Error desconocido al analizar el video.") if datos else "No se obtuvieron datos."
            st.error(f"Error: {mensaje_error}")
            st.stop()

        # Recodificación a H.264 para compatibilidad en el reproductor web
        video_web_listo = False
        if os.path.exists(ruta_tmp_out):
            try:
                clip = VideoFileClip(ruta_tmp_out)
                clip.write_videofile(
                    ruta_tmp_web,
                    codec="libx264",
                    audio=False,
                    logger=None,
                )
                clip.close()
                video_web_listo = os.path.exists(ruta_tmp_web)
            except Exception as exc:
                st.warning(f"No se pudo recodificar el video para vista web: {exc}")
            finally:
                if os.path.exists(ruta_tmp_out):
                    try:
                        os.unlink(ruta_tmp_out)
                    except OSError:
                        pass

        modelo_usado = datos.get("modelo_usado", "yolo11n.pt")
        st.caption(f"Modelo de detección utilizado: `{modelo_usado}`")

        resumen = datos["resumen"]
        tiros = datos["tiros"]
        recomendaciones = datos.get("recomendaciones", [])

        # Resumen de métricas
        st.subheader("📊 Resumen del análisis")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("🏀 Tiros evaluados", resumen["tiros_detectados"])
        col2.metric("✅ Tiros correctos", resumen["tiros_correctos"])
        col3.metric("⭐ Puntaje", f"{resumen['puntaje_promedio']}/100")
        col4.metric("⚡ Velocidad máx.", f"{resumen['velocidad_maxima']} km/h")

        st.divider()

        # Reproducción de videos
        st.subheader("🎥 Comparación visual")
        tab_procesado, tab_original = st.tabs(["Video Procesado", "Video Original"])

        with tab_procesado:
            if video_web_listo:
                st.video(ruta_tmp_web)
            else:
                st.info("No hay vista previa procesada disponible para este archivo.")

        with tab_original:
            st.video(archivo_video)

        st.divider()

        # Gráfica de puntaje y ángulos
        col_graf, col_det = st.columns([1.2, 1])

        with col_graf:
            st.subheader("🎯 Puntaje técnico")
            fig = go.Figure()
            fig.add_trace(
                go.Bar(
                    x=[f"Tiro {t['id']}" for t in tiros],
                    y=[t["puntaje"] for t in tiros],
                    text=[f"{t['puntaje']} pts" for t in tiros],
                    textposition="auto",
                    marker_color=["#00CC96" if t["correcto"] else "#EF553B" for t in tiros],
                )
            )
            fig.add_hline(
                y=80,
                line_dash="dash",
                line_color="green",
                annotation_text="Meta (80 pts)",
            )
            fig.update_layout(
                yaxis=dict(range=[0, 105]),
                height=340,
                margin=dict(l=20, r=20, t=30, b=20),
            )
            st.plotly_chart(fig, use_container_width=True)

        with col_det:
            st.subheader("📐 Ángulos registrados")
            for t in tiros:
                st.write(f"- **Flexión mínima de rodilla:** {t['flexion_rodilla']}°")
                st.write(f"- **Extensión máxima de codo:** {t['extension_codo']}°")
                st.write(f"- **Ángulo de hombro:** {t['angulo_hombro']}°")

            st.subheader("💡 Retroalimentación")
            for rec in recomendaciones:
                if "Excelente" in rec:
                    st.success(rec)
                else:
                    st.warning(rec)
else:
    st.info("👈 Sube un video grabado en el panel izquierdo para comenzar el análisis.")