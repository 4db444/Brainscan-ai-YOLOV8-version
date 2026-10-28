import json
from pathlib import Path

from numpy import ndarray, save, lib


def save_npy (array : ndarray, path : Path, name : str) -> None:
    path.mkdir(parents=True, exist_ok=True)
    save(path / f"{name}.npy", array)


def save_json (data : dict, path : Path, name : str) -> None:
    path.mkdir(parents=True, exist_ok=True)
    with open(path / f"{name}.json", "w") as file:
        json.dump(data, file, indent=4)


def _save_normalized_npy (x : ndarray, path : Path, name : str,
                          chunk_size : int = 256) -> None:
    # Normalize [0, 1] row by row straight into the file: constant RAM
    path.mkdir(parents=True, exist_ok=True)
    target = path / f"{name}.npy"
    memmap = lib.format.open_memmap(
        target, mode="w+", dtype="float32", shape=x.shape,
    )
    for start in range(0, len(x), chunk_size):
        chunk = x[start:start + chunk_size]
        memmap[start:start + chunk_size] = chunk.astype("float32") / 255.0
    memmap.flush()
    del memmap


def save_processed_data (dataset : dict, path : Path) -> None:
    _save_normalized_npy(dataset["x_train"], path, "x_train")
    _save_normalized_npy(dataset["x_test"], path, "x_test")
    save_npy(dataset["y_train"], path, "y_train")
    save_npy(dataset["y_test"], path, "y_test")
    save_json({"class_names": dataset["class_names"]}, path, "class_names")
    print(f"Processed data saved to {path}")
