import base64
import io

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from ai.dataset import IMAGE_DIR, MASK_DIR, TOTAL, generate_dataset
from ai.process import apply_morph, to_binary


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


def make_roc(images, masks):
    tpr_list = []
    fpr_list = []
    for thresh in range(0, 256, 8):
        tp = fp = tn = fn = 0
        for image, mask in zip(images, masks):
            pred = np.where(image >= thresh, 255, 0).astype(np.uint8)
            pred = apply_morph(pred, "opening")
            s = scores(pred, mask)
            tp += s["tp"]
            fp += s["fp"]
            tn += s["tn"]
            fn += s["fn"]
        tpr_list.append(tp / (tp + fn) if (tp + fn) else 0)
        fpr_list.append(fp / (fp + tn) if (fp + tn) else 0)

    fig, ax = plt.subplots(figsize=(4, 4))
    ax.plot(fpr_list, tpr_list, color="#6b4f3a")
    ax.plot([0, 1], [0, 1], color="#c4a484", linestyle="--")
    ax.set_xlabel("FPR")
    ax.set_ylabel("TPR")
    ax.set_title("ROC Curve")
    fig.patch.set_facecolor("#e8d8c4")
    ax.set_facecolor("#dfcbb3")
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("ascii")


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
        opened = apply_morph(binary, "opening")
        a = scores(binary, mask)
        b = scores(opened, mask)
        for key in without:
            without[key] += a[key]
            with_morph[key] += b[key]

    return {
        "success": True,
        "total": len(images),
        "without_morph": metrics(**without),
        "with_morph": metrics(**with_morph),
        "roc_image": make_roc(images, masks),
        "note": "object = Positive, background = Negative",
    }
