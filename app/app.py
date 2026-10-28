from pathlib import Path
from tempfile import NamedTemporaryFile
import shutil

from streamlit import set_page_config, title, file_uploader, image, subheader, metric, error, caption
from tensorflow.keras.models import load_model

from src.config import FINAL_MODEL_PATH
from src.models import predict_image

set_page_config(page_title="BrainScan AI", page_icon="🧠", layout="centered")
title("🧠 BrainScan AI — Tumor MRI Classification")

model = load_model(FINAL_MODEL_PATH)
caption(f"Model loaded from `{FINAL_MODEL_PATH.name}`")

uploaded_file = file_uploader("Upload an MRI scan", type=["jpg", "jpeg", "png", "bmp"])

if uploaded_file is not None:
    with NamedTemporaryFile(delete=False, suffix=Path(uploaded_file.name).suffix) as tmp_file:
        tmp_file.write(uploaded_file.getbuffer())
        tmp_path = Path(tmp_file.name)

    try:
        image(str(tmp_path), width=320)
        result = predict_image(tmp_path)
        subheader(f"Prediction: **{result['class_name']}**")
        metric("Confidence", f"{result['confidence'] * 100:.1f}%")
        subheader("Probabilities")
        for class_name, probability in result["probabilities"].items():
            metric(class_name, f"{probability * 100:.1f}%")
    except Exception as exc:
        error(f"An error occurred: {exc}")
    finally:
        tmp_path.unlink(missing_ok=True)
