# HoopCoach AI

<h2 align="center">
  Biomechanical posture and free-throw shooting analysis tool for basketball training videos
</h2>

Hi! We are **Chanty4430**, and we created **HoopCoach AI** as an educational project. The goal is to build a practical tool that helps basketball players refine their posture and shooting mechanics—especially free throws—from recorded individual training sessions while respecting each player's natural shooting style.

You can find us on the Hack Club Slack as: `@Chanty_30`

**HoopCoach AI** is a Python-based basketball posture and mechanics analyzer that uses Computer Vision and Machine Learning to evaluate recorded training videos (`MP4`, `MOV`, `AVI`). Rather than running live real-time camera tracking, the system processes pre-recorded clips to extract joint angles with MediaPipe, visualize biomechanical feedback in Streamlit, and train tabular ML models on angular metrics from shooting sessions and elite free-throw reference clips.

---

## Libraries and Tools Used

| Technology | Function |
| :--- | :--- |
| **Python** | Main programming language |
| **OpenCV** | Video reading, lighting normalization (CLAHE), frame rotation handling, and rendering |
| **MediaPipe** | Human pose estimation and joint landmark extraction |
| **NumPy & Pandas** | Numerical computations, angle percentiles, and dataset handling |
| **Streamlit & Plotly** | Interactive web dashboard for video playback, metrics, and angle charts |
| **scikit-learn & joblib** | Tabular Machine Learning training, classification, and model persistence |
| **MoviePy** | Video re-encoding to H.264 for browser playback compatibility |

---

## Project Structure

- `app/`: Streamlit interface (`streamlit_app.py`), video posture analyzer (`analizador_video.py`), and biomechanical evaluation logic (`biomecanica.py`).
- `training/`: Scripts and datasets for extracting free-throw angles (`extraer_angulos.py`), training ML models (`entrenar_ml.py`), and testing predictions (`probar_ml.py`).
- `models/`: Saved trained models (such as `modelo_tiro_libre.joblib`).
- `samples/`: Recorded training and reference free-throw clips.

---

## Current Status

HoopCoach AI currently supports:

- Analyzing recorded individual shooting videos (no live webcam / real-time stream required).
- Extracting joint angles (knee flexion, elbow extension, and shoulder angles) using MediaPipe pose landmarks.
- Normalizing lighting with CLAHE and rendering processed videos with pose skeletons in H.264 for Streamlit playback.
- Evaluating shooting posture and displaying scores, angle summaries, areas to improve, and recommendations.
- Extracting angular features into a CSV dataset (`dataset_tiros.csv`), training a `scikit-learn` model (`entrenar_ml.py`), and running inference/comparisons (`probar_ml.py`).

### Work in Progress & Next Steps

- Expanding the free-throw angle dataset with clips from individual training sessions and elite free-throw shooters (such as Stephen Curry, Steve Nash, Mark Price, Rick Barry, and Damian Lillard) while accounting for distinct shooting mechanics.
- Refining angle extraction and rotation handling for vertical or rotated phone videos.
- Integrating trained ML style/reference comparisons more closely into the analysis workflow.
- Re-introducing optional ball detection/speed modules later once posture and ML mechanics are solid.

---

## License
