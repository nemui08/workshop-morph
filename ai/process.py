import cv2
import numpy as np


def to_binary(gray):
    _unused, binary = cv2.threshold(
        gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    if np.mean(binary) > 127:
        binary = 255 - binary
    return binary


def apply_morph(binary, model):
    se = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))

    if model == "erosion":
        return cv2.erode(binary, se)
    if model == "opening":
        return cv2.morphologyEx(binary, cv2.MORPH_OPEN, se)
    if model == "closing":
        return cv2.morphologyEx(binary, cv2.MORPH_CLOSE, se)
    return cv2.dilate(binary, se)


def process(image_bytes, user_id, model="dilation"):
    arr = np.frombuffer(image_bytes, dtype=np.uint8)
    gray = cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE)
    binary = to_binary(gray)
    out = apply_morph(binary, model)
    _ok, buf = cv2.imencode(".jpg", out)

    return {
        "user_id": user_id,
        "status": "ok",
        "note": "(Demo) " + model,
        "image_bytes": buf.tobytes(),
        "mimetype": "image/jpeg",
    }
