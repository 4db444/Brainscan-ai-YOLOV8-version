import shutil
import zipfile
from pathlib import Path

import gdown

from src.config import RAW_DATA_DIR, TMP_DIR


def download_zip() -> Path:
    """Download the zip from Google Drive (skipped if already downloaded)."""
    TMP_DIR.mkdir(parents=True, exist_ok=True)
    zip_path = TMP_DIR / "brain_tumor_mri.zip"

    if zip_path.exists():
        print(f"Zip already downloaded: {zip_path}")
        return zip_path

    url = f"https://drive.google.com/file/d/1WtjAJx4k_r2dI4YF4Ht84LcGDoerltQY/view"
    result = gdown.download(url, str(zip_path), quiet=False)
    if result is None:
        raise RuntimeError(
            "Download failed. Check that the file is shared as 'Anyone with the link' "
            "this file; try again later."
        )
    return zip_path


def extract_zip(zip_path: Path) -> Path:
    """Extract the zip into a temp folder and return the real dataset root."""
    extract_dir = TMP_DIR / "extracted"
    if extract_dir.exists():
        shutil.rmtree(extract_dir)

    try:
        with zipfile.ZipFile(zip_path) as zf:
            corrupted = zf.testzip()
            if corrupted:
                raise zipfile.BadZipFile(f"Corrupted file in archive: {corrupted}")
            zf.extractall(extract_dir)
    except zipfile.BadZipFile:
        zip_path.unlink(missing_ok=True)   # force a fresh download next run
        raise

    # Many zips wrap everything in one top-level folder: unwrap it
    root = extract_dir
    entries = [p for p in root.iterdir() if p.name != "__MACOSX"]
    while len(entries) == 1 and entries[0].is_dir():
        root = entries[0]
        entries = list(root.iterdir())
    return root


def main(force: bool = False) -> None:
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)

    if any(p for p in RAW_DATA_DIR.iterdir() if p.name != ".gitkeep") and not force:
        print("Data already present in data/raw/, skipping. Use --force to redo.")
        return

    zip_path = download_zip()
    dataset_root = extract_zip(zip_path)

    for item in dataset_root.iterdir():
        target = RAW_DATA_DIR / item.name
        if target.exists():
            shutil.rmtree(target) if target.is_dir() else target.unlink()
        shutil.move(str(item), str(target))

    shutil.rmtree(TMP_DIR, ignore_errors=True)   # remove zip + temp files
    print(f"Dataset ready in {RAW_DATA_DIR}")
    print("Classes found:", sorted(p.name for p in RAW_DATA_DIR.iterdir() if p.is_dir()))


if __name__ == "__main__":
    import sys
    main(force="--force" in sys.argv)