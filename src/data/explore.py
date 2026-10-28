from collections import Counter

from matplotlib import pyplot
from numpy import ndarray

from ..config import REPORTS_FIGURES_DIR


def count_images_per_class (y : ndarray) -> dict[str, int]:
    # Guard against empty input
    if y.size == 0:
        print("Warning: empty label array – no images to count.")
        return {}
    counts = dict(Counter(y.tolist()))
    print("Images per class:")
    for class_name, count in counts.items():
        print(f"  {class_name}: {count}")
    return counts


def check_class_balance (y : ndarray, threshold : float = 0.1) -> bool:
    # Guard against empty input – treat as balanced (nothing to rebalance)
    if y.size == 0:
        print("Warning: empty label array – cannot assess balance; treating as balanced.")
        return True
    counts = list(Counter(y.tolist()).values())
    if not counts:
        print("Warning: no class counts found – treating as balanced.")
        return True
    imbalance_ratio = (max(counts) - min(counts)) / max(counts)
    balanced = imbalance_ratio <= threshold
    print(f"Imbalance ratio: {imbalance_ratio:.3f} -> {'balanced' if balanced else 'imbalanced'}")
    return balanced


def show_sample_images (x : ndarray, y : ndarray, class_names : list[str],
                        samples_per_class : int = 5) -> None:
    figure, axes = pyplot.subplots(len(class_names), samples_per_class,
                                   figsize=(samples_per_class * 2, len(class_names) * 2))
    for row, class_name in enumerate(class_names):
        indices = [i for i, label in enumerate(y) if label == class_name][:samples_per_class]
        for col in range(samples_per_class):
            axis = axes[row, col]
            axis.axis("off")
            if col < len(indices):
                axis.imshow(x[indices[col]].astype("uint8"))
                if col == 0:
                    axis.set_title(class_name)
    pyplot.tight_layout()
    REPORTS_FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    pyplot.savefig(REPORTS_FIGURES_DIR / "sample_images.png")
    pyplot.show()
