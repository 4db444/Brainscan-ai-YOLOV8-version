from numpy import ndarray, argmax
from pandas import DataFrame
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from tensorflow.keras.models import Model

from ..config import REPORTS_TABLES_DIR
from ..data.save_data import save_json


def evaluate_model (model : Model, x_test : ndarray, y_test : ndarray,
                    class_names : list[str]) -> dict:
    loss, accuracy = model.evaluate(x_test, y_test, verbose=0)
    print(f"Test loss: {loss:.4f}, Test accuracy: {accuracy:.4f}")

    y_true = argmax(y_test, axis=1)
    y_pred = argmax(model.predict(x_test, verbose=0), axis=1)

    report_dict = classification_report(y_true, y_pred,
                                        target_names=class_names, output_dict=True)
    report_text = classification_report(y_true, y_pred, target_names=class_names)
    print(report_text)

    REPORTS_TABLES_DIR.mkdir(parents=True, exist_ok=True)
    DataFrame(report_dict).to_csv(REPORTS_TABLES_DIR / "classification_report.csv")
    save_json(
        {
            "accuracy": float(accuracy),
            "loss": float(loss),
            "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
        },
        REPORTS_TABLES_DIR,
        "evaluation_stats",
    )

    return {
        "name": "cnn",
        "stats": {"loss": float(loss), "accuracy": float(accuracy)},
        "y_true": y_true,
        "y_pred": y_pred,
        "report": report_dict,
    }
