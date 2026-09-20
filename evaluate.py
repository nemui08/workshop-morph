import zipfile
from pathlib import Path
from urllib.request import urlopen

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent
IMAGE_DIR = ROOT / "dataset" / "images"
MASK_DIR = ROOT / "dataset" / "masks"
TOTAL = 50
SIZE = 256
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


def generate_dataset():
    IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    MASK_DIR.mkdir(parents=True, exist_ok=True)
    raw_dir = ROOT / "dataset" / "raw"
    images_zip = _download(IMAGES_ZIP, raw_dir / "images.zip")
    masks_zip = _download(MASKS_ZIP, raw_dir / "masks.zip")
    with zipfile.ZipFile(images_zip) as iz, zipfile.ZipFile(masks_zip) as mz:
        image_map = {
            _key(n): n
            for n in iz.namelist()
            if n.lower().endswith((".jpg", ".jpeg", ".png"))
        }
        mask_map = {
            _key(n): n
            for n in mz.namelist()
            if n.lower().endswith((".jpg", ".jpeg", ".png"))
        }
        names = sorted(set(image_map) & set(mask_map))[:TOTAL]
        for i, stem in enumerate(names):
            image = cv2.imdecode(
                np.frombuffer(iz.read(image_map[stem]), dtype=np.uint8),
                cv2.IMREAD_UNCHANGED,
            )
            mask = cv2.imdecode(
                np.frombuffer(mz.read(mask_map[stem]), dtype=np.uint8),
                cv2.IMREAD_UNCHANGED,
            )
            image = cv2.resize(image, (SIZE, SIZE))
            mask = cv2.resize(mask, (SIZE, SIZE), interpolation=cv2.INTER_NEAREST)
            if mask.ndim == 3:
                mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
            _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
            name = f"{i:03d}.png"
            cv2.imwrite(str(IMAGE_DIR / name), image)
            cv2.imwrite(str(MASK_DIR / name), mask)
    return len(names)


def to_binary(gray):
    _unused, binary = cv2.threshold(
        gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    if np.mean(binary) > 127:
        binary = 255 - binary
    return binary


def opening(binary):
    se = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    return cv2.morphologyEx(binary, cv2.MORPH_OPEN, se)


def metrics(tp, fp, tn, fn):
    acc = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp) if (tp + fp) else 0
    recall = tp / (tp + fn) if (tp + fn) else 0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall)
        else 0
    )
    return {
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn),
        "accuracy": round(acc, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
    }


def scores(pred, truth):
    pred_pos = pred > 127
    truth_pos = truth > 127
    tp = int(np.sum(pred_pos & truth_pos))
    fp = int(np.sum(pred_pos & ~truth_pos))
    tn = int(np.sum(~pred_pos & ~truth_pos))
    fn = int(np.sum(~pred_pos & truth_pos))
    return metrics(tp, fp, tn, fn)


def roc_points(images, masks):
    tpr_list = []
    fpr_list = []
    for thresh in range(0, 256, 8):
        tp = fp = tn = fn = 0
        for image, mask in zip(images, masks):
            # coins are darker than sand → higher score = more coin-like
            pred = np.where((255 - image) >= thresh, 255, 0).astype(np.uint8)
            pred = opening(pred)
            s = scores(pred, mask)
            tp += s["tp"]
            fp += s["fp"]
            tn += s["tn"]
            fn += s["fn"]
        tpr_list.append(tp / (tp + fn) if (tp + fn) else 0)
        fpr_list.append(fp / (fp + tn) if (fp + tn) else 0)
    return fpr_list, tpr_list


def evaluate():
    if not IMAGE_DIR.exists() or len(list(IMAGE_DIR.glob("*.png"))) < TOTAL:
        generate_dataset()

    images = []
    masks = []
    without = {"tp": 0, "fp": 0, "tn": 0, "fn": 0}
    with_morph = {"tp": 0, "fp": 0, "tn": 0, "fn": 0}

    for path in sorted(IMAGE_DIR.glob("*.png")):
        image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(str(MASK_DIR / path.name), cv2.IMREAD_GRAYSCALE)
        images.append(image)
        masks.append(mask)

        binary = to_binary(image)
        opened = opening(binary)
        a = scores(binary, mask)
        b = scores(opened, mask)
        for key in without:
            without[key] += a[key]
            with_morph[key] += b[key]

    fpr_list, tpr_list = roc_points(images, masks)
    return {
        "success": True,
        "total": len(images),
        "without_morph": metrics(**without),
        "with_morph": metrics(**with_morph),
        "fpr": fpr_list,
        "tpr": tpr_list,
        "note": "object = Positive, background = Negative",
    }
