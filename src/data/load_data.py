from pathlib import Path

from cv2 import imread, resize
from numpy import ndarray, array
from numpy.random import default_rng

from ..config import RAW_DATA_DIR, VALID_EXTENSIONS, IMAGE_SIZE, RANDOM_SEED


def _list_classes (data_dir : Path) -> list[str]:
    return sorted(
        folder.name for folder in data_dir.iterdir()
        if folder.is_dir()
    )


def _is_valid_image (path : Path) -> bool:
    return path.suffix.lower() in VALID_EXTENSIONS


def _load_image (path : Path) -> ndarray | None:
    try:
        image = imread(str(path))
        if image is None:
            return None
        return resize(image, (IMAGE_SIZE, IMAGE_SIZE))
    except Exception:
        return None


def load_data (data_dir : Path = RAW_DATA_DIR / "Train") -> tuple[ndarray, ndarray, list[str]]:
    class_names = _list_classes(data_dir)
    images : list[ndarray] = []
    labels : list[str] = []

    for class_name in class_names:
        class_dir = data_dir / class_name
        # Use the nested "images" subdirectory if present
        img_dir = class_dir / "images" if (class_dir / "images").is_dir() else class_dir
        for path in sorted(img_dir.iterdir()):
            if not _is_valid_image(path):
                continue
            image = _load_image(path)
            if image is None:
                print(f"Skipping unreadable image: {path}")
                continue
            images.append(image)
            labels.append(class_name)

    # Shuffle images and labels together, same index in both lists
    rng = default_rng(RANDOM_SEED)
    order = rng.permutation(len(images))
    images = [images[i] for i in order]
    labels = [labels[i] for i in order]

    x = array(images)
    y = array(labels)
    print(f"Loaded {len(labels)} images across {len(class_names)} classes")
    print(f"X shape: {x.shape}, y shape: {y.shape}")
    return x, y, class_names
