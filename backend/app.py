import base64
import sys
from pathlib import Path

from fastapi import FastAPI, File, Form, UploadFile
from fastapi.middleware.cors import CORSMiddleware

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from ai.evaluate import evaluate
from ai.process import process


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/api/process")
async def api_process(
    file: UploadFile = File(...),
    user_id: int = Form(1),
    model: str = Form("dilation"),
):
    data = await file.read()

    result = process(data, user_id, model)

    return {
        "status": result["status"],
        "note": result["note"],
        "model": model,
        "image": base64.b64encode(result["image_bytes"]).decode("ascii"),
        "mimetype": result.get("mimetype") or "image/jpeg",
    }


@app.get("/api/evaluate")
def api_evaluate():
    return evaluate()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8001,
        reload=True,
        reload_dirs=[str(ROOT)],
    )
