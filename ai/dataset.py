import zipfile
from pathlib import Path
from urllib.request import urlopen

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
IMAGE_DIR = ROOT / "dataset" / "images"
MASK_DIR = ROOT / "dataset" / "masks"
TOTAL = 50
SIZE = 256

# Labeled Images of Sand and Coins v2 · Buscombe 2022 · CC BY 4.0
# https://doi.org/10.5281/zenodo.6232246
IMAGES_ZIP = "https://zenodo.org/api/records/6232246/files/images.zip/content"
MASKS_ZIP = "https://zenodo.org/api/records/6232246/files/masks.zip/content"


def _download(url, dest):
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    with urlopen(url, timeout=180) as res, open(dest, "wb") as f:
        while True:
            chunk = res.read(1024 * 1024)
            if not chunk:
                break
            f.write(chunk)
    return dest


def _key(name):
    stem = Path(name).stem
    if stem.endswith("_mask"):
        stem = stem[:-5]
    return stem


def _zip_map(zf):
    out = {}
    for name in zf.namelist():
        low = name.lower()
        if low.endswith((".jpg", ".jpeg", ".png")):
            out[_key(name)] = name
    return out


def _decode(zf, name):
    arr = np.frombuffer(zf.read(name), dtype=np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_UNCHANGED)


def generate_dataset():
    IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    MASK_DIR.mkdir(parents=True, exist_ok=True)

    raw_dir = ROOT / "dataset" / "raw"
    images_zip = _download(IMAGES_ZIP, raw_dir / "images.zip")
    masks_zip = _download(MASKS_ZIP, raw_dir / "masks.zip")

    with zipfile.ZipFile(images_zip) as iz, zipfile.ZipFile(masks_zip) as mz:
        image_map = _zip_map(iz)
        mask_map = _zip_map(mz)
        names = sorted(set(image_map) & set(mask_map))[:TOTAL]

        for i, stem in enumerate(names):
            image = _decode(iz, image_map[stem])
            mask = _decode(mz, mask_map[stem])
            image = cv2.resize(image, (SIZE, SIZE))
            mask = cv2.resize(mask, (SIZE, SIZE), interpolation=cv2.INTER_NEAREST)
            if mask.ndim == 3:
                mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
            _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
            name = f"{i:03d}.png"
            cv2.imwrite(str(IMAGE_DIR / name), image)
            cv2.imwrite(str(MASK_DIR / name), mask)

    (ROOT / "dataset" / "SOURCE.txt").write_text(
        "Labeled Images of Sand and Coins v2\n"
        "Buscombe, Daniel (2022)\n"
        "https://doi.org/10.5281/zenodo.6232246\n"
        "CC BY 4.0\n"
        "coin = Positive (white), sand/background = Negative (black)\n",
        encoding="utf-8",
    )
    return len(names)


if __name__ == "__main__":
    print(generate_dataset())
