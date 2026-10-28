from matplotlib import pyplot
from numpy import ndarray
from seaborn import heatmap
from sklearn.metrics import confusion_matrix


def plot_class_distribution (counts : dict[str, int], path) -> None:
    pyplot.figure(figsize=(8, 5))
    pyplot.bar(counts.keys(), counts.values(), color="steelblue")
    pyplot.title("Images per class")
    pyplot.xlabel("Class")
    pyplot.ylabel("Count")
    pyplot.xticks(rotation=15)
    pyplot.tight_layout()
    path.mkdir(parents=True, exist_ok=True)
    pyplot.savefig(path / "class_distribution.png")
    pyplot.show()


def plot_learning_curves (history : dict, path) -> None:
    figure, (accuracy_axis, loss_axis) = pyplot.subplots(1, 2, figsize=(12, 4))

    accuracy_axis.plot(history["accuracy"], label="train")
    accuracy_axis.plot(history["val_accuracy"], label="val")
    accuracy_axis.set_title("Model accuracy")
    accuracy_axis.set_xlabel("Epoch")
    accuracy_axis.legend()

    loss_axis.plot(history["loss"], label="train")
    loss_axis.plot(history["val_loss"], label="val")
    loss_axis.set_title("Model loss")
    loss_axis.set_xlabel("Epoch")
    loss_axis.legend()

    pyplot.tight_layout()
    path.mkdir(parents=True, exist_ok=True)
    pyplot.savefig(path / "learning_curves.png")
    pyplot.show()


def plot_confusion_matrix (y_true : ndarray, y_pred : ndarray,
                           class_names : list[str], path) -> None:
    matrix = confusion_matrix(y_true, y_pred)
    pyplot.figure(figsize=(7, 6))
    heatmap(matrix, annot=True, fmt="d", cmap="Blues",
            xticklabels=class_names, yticklabels=class_names)
    pyplot.title("Confusion matrix")
    pyplot.xlabel("Predicted")
    pyplot.ylabel("True")
    pyplot.tight_layout()
    path.mkdir(parents=True, exist_ok=True)
    pyplot.savefig(path / "confusion_matrix.png")
    pyplot.show()


def plot_prediction_examples (x : ndarray, y_true : ndarray, y_pred : ndarray,
                              class_names : list[str], path, samples : int = 8) -> None:
    correct = [i for i in range(len(y_true)) if y_true[i] == y_pred[i]][:samples // 2]
    incorrect = [i for i in range(len(y_true)) if y_true[i] != y_pred[i]][:samples // 2]
    indices = correct + incorrect

    figure, axes = pyplot.subplots(2, len(indices) // 2, figsize=(len(indices) * 1.5, 4))
    axes = axes.ravel()
    for axis, index in zip(axes, indices):
        is_correct = y_true[index] == y_pred[index]
        axis.imshow(x[index])
        axis.set_title(
            f"{'OK' if is_correct else 'KO'}\n{class_names[y_true[index]]}\n{class_names[y_pred[index]]}",
            color="green" if is_correct else "red",
            fontsize=8,
        )
        axis.axis("off")
    pyplot.tight_layout()
    path.mkdir(parents=True, exist_ok=True)
    pyplot.savefig(path / "prediction_examples.png")
    pyplot.show()
