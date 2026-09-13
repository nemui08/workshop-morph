from pathlib import Path

from flask import Flask, jsonify, render_template, send_from_directory
import requests

DATASET_DIR = Path(__file__).resolve().parent.parent / "dataset"

app = Flask(__name__)
# ต้องใส่เป็น url ของ backend
BACKEND_URL = "http://127.0.0.1:8001"
# BACKEND_URL = "http://172.20.56.115:8001"


@app.route("/")
def home():
    samples = [p.name for p in sorted((DATASET_DIR / "images").glob("*.png"))]
    return render_template("home.html", samples=samples, total=len(samples))


@app.route("/dataset/<kind>/<name>")
def dataset_file(kind, name):
    if kind not in ("images", "masks"):
        return "", 404
    return send_from_directory(DATASET_DIR / kind, name)


@app.route("/api/evaluate")
def evaluate_workshop():
    try:
        response = requests.get(f"{BACKEND_URL}/api/evaluate", timeout=60)
        return jsonify(response.json())
    except requests.RequestException:
        return jsonify({"success": False, "message": "ไม่สามารถเชื่อมต่อ Backend ได้"}), 500


if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=5001,
        debug=True,
    )
