from collections import Counter

from numpy import ndarray, random, eye
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from ..config import RANDOM_SEED, TEST_SIZE


def rebalance_data (x : ndarray, y : ndarray) -> tuple[ndarray, ndarray]:
    # Undersample every class down to the size of the smallest one
    counts = Counter(y.tolist())
    min_count = min(counts.values())
    rng = random.default_rng(RANDOM_SEED)

    indices = []
    for class_name in counts:
        class_indices = [i for i, label in enumerate(y) if label == class_name]
        picked = rng.choice(class_indices, size=min_count, replace=False)
        indices.extend(picked.tolist())

    indices = rng.permutation(indices).tolist()
    print(f"Rebalanced to {min_count} images per class ({len(indices)} total)")
    return x[indices], y[indices]


def encode_labels (y : ndarray) -> tuple[ndarray, LabelEncoder, list[str]]:
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    class_names = list(label_encoder.classes_)
    print("Encoded labels:", dict(zip(class_names, range(len(class_names)))))
    return y_encoded, label_encoder, class_names


def normalize_images (x : ndarray) -> ndarray:
    return x.astype("float32") / 255.0


def preprocess_data (x : ndarray, y : ndarray) -> dict:
    y_encoded, label_encoder, class_names = encode_labels(y)

    # Images stay uint8 here; normalization [0, 1] happens chunk-wise at save
    # time (see save_processed_data) to keep peak RAM low
    x_train, x_test, y_train, y_test = train_test_split(
        x, y_encoded,
        test_size=TEST_SIZE,
        random_state=RANDOM_SEED,
        stratify=y_encoded,
    )

    # One-hot targets for categorical_crossentropy
    y_train = eye(len(class_names), dtype="float32")[y_train]
    y_test = eye(len(class_names), dtype="float32")[y_test]

    print(f"Train: {x_train.shape}, Test: {x_test.shape}")
    return {
        "x_train": x_train,
        "x_test": x_test,
        "y_train": y_train,
        "y_test": y_test,
        "class_names": class_names,
    }
