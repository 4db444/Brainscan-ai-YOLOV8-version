from time import time

from numpy import ndarray
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.models import Model

from ..config import BATCH_SIZE, EPOCHS, MODELS_DIR, FINAL_MODEL_PATH


def _build_callbacks () -> list:
    return [
        # Keep the best model on val_accuracy, not the last epoch
        ModelCheckpoint(
            filepath=MODELS_DIR / "best_model.h5",
            monitor="val_accuracy",
            save_best_only=True,
            verbose=1,
        ),
        EarlyStopping(
            monitor="val_accuracy",
            patience=5,
            restore_best_weights=True,
            verbose=1,
        ),
    ]


def train_model (model : Model, x_train : ndarray, y_train : ndarray,
                 x_test : ndarray, y_test : ndarray) -> dict:
    start = time()
    history = model.fit(
        x_train, y_train,
        validation_data=(x_test, y_test),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=_build_callbacks(),
        shuffle=True,
        verbose=1,
    )
    training_time = time() - start
    print(f"Training finished in {training_time:.1f}s")

    model.save(FINAL_MODEL_PATH)
    print(f"Final model saved to {FINAL_MODEL_PATH}")

    return {
        "name": "cnn",
        "model": model,
        "history": history.history,
        "training_time": training_time,
    }
