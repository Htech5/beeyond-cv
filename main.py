from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO
from PIL import Image
import io

app = FastAPI()

# Izinkan request dari Next.js dev server
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # nanti persempit ke domain kamu saat production
    allow_methods=["*"],
    allow_headers=["*"],
)

model = YOLO("yolov8n.pt")

@app.post("/detect")
async def detect(file: UploadFile = File(...)):
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")

    results = model(image)

    detections = []
    for r in results:
        for box in r.boxes:
            class_id = int(box.cls[0])
            label = model.names[class_id]
            confidence = float(box.conf[0])
            detections.append({"label": label, "confidence": round(confidence, 3)})

    # Ambil deteksi dengan confidence tertinggi (paling yakin)
    best = max(detections, key=lambda d: d["confidence"]) if detections else None

    return {"detections": detections, "best": best}