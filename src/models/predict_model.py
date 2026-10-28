import json
from pathlib import Path

from cv2 import imread, resize
from numpy import ndarray, expand_dims, argmax
from tensorflow.keras.models import load_model

from ..config import IMAGE_SIZE, FINAL_MODEL_PATH, CLASS_NAMES_PATH


def _load_class_names () -> list[str]:
    with open(CLASS_NAMES_PATH) as file:
        return json.load(file)["class_names"]


def predict_image (path : Path, model_path : Path | None = None) -> dict:
    # Default resolved at call time so tests/scripts can redirect FINAL_MODEL_PATH
    model_path = model_path or FINAL_MODEL_PATH
    model = load_model(model_path)
    class_names = _load_class_names()

    # Same preprocessing as training: resize then normalize to [0, 1]
    image = imread(str(path))
    if image is None:
        raise ValueError(f"Cannot read image: {path}")
    image = resize(image, (IMAGE_SIZE, IMAGE_SIZE))
    image = expand_dims(image.astype("float32") / 255.0, axis=0)

    probabilities = model.predict(image, verbose=0)[0]
    best_index = int(argmax(probabilities))

    return {
        "class_name": class_names[best_index],
        "confidence": float(probabilities[best_index]),
        "probabilities": {
            class_name: float(probability)
            for class_name, probability in zip(class_names, probabilities)
        },
    }
