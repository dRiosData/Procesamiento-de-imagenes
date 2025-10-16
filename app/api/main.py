# app/api/main.py
from __future__ import annotations

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from PIL import Image, UnidentifiedImageError
import numpy as np
import io
import base64
import cv2
from typing import List

from app.detection.yolov5_detector import OnnxYoloV5Detector

# -----------------------------------------------------------------------------
# App & middleware
# -----------------------------------------------------------------------------
app = FastAPI(title="Detector de Seguridad - MVP")

# CORS (en producción restringí allow_origins a tu dominio)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Servir la app web estática
app.mount("/web", StaticFiles(directory="web", html=True), name="web")

# Cargar modelo una sola vez
DETECTOR = OnnxYoloV5Detector(model_path="models/yolov5s.onnx", img_size=640)

# -----------------------------------------------------------------------------
# Utilidades de dibujo
# -----------------------------------------------------------------------------
COCO_NAMES = [
    "person","bicycle","car","motorbike","aeroplane","bus","train","truck","boat","traffic light",
    "fire hydrant","stop sign","parking meter","bench","bird","cat","dog","horse","sheep","cow",
    "elephant","bear","zebra","giraffe","backpack","umbrella","handbag","tie","suitcase",
]

def draw_detections(img_bgr: np.ndarray, dets: List[dict]) -> np.ndarray:
    """Dibuja cajas + etiquetas sobre una imagen BGR y devuelve una copia."""
    out = img_bgr.copy()
    for d in dets:
        x1, y1, x2, y2 = map(int, d["box"])
        cls_id, score = d["cls"], d["score"]
        label = COCO_NAMES[cls_id] if 0 <= cls_id < len(COCO_NAMES) else str(cls_id)

        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 0), 2)
        txt = f"{label} {score:.2f}"
        (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(out, (x1, y1 - th - 6), (x1 + tw + 4, y1), (0, 255, 0), -1)
        cv2.putText(out, txt, (x1 + 2, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    return out

def draw_boxes(image_bgr: np.ndarray, detections: List[dict]) -> bytes:
    """Dibuja cajas y devuelve la imagen codificada en PNG (bytes)."""
    img = image_bgr.copy()
    for det in detections:
        x1, y1, x2, y2 = map(int, det["box"])
        score = det["score"]
        cls_id = det["cls"]
        name = COCO_NAMES[cls_id] if 0 <= cls_id < len(COCO_NAMES) else str(cls_id)
        label = f"{name} {score:.2f}"

        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(img, (x1, y1 - th - 8), (x1 + tw + 6, y1), (0, 255, 0), -1)
        cv2.putText(img, label, (x1 + 3, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

    ok, png = cv2.imencode(".png", img)
    if not ok:
        raise RuntimeError("No se pudo codificar PNG")
    return png.tobytes()

# -----------------------------------------------------------------------------
# Endpoints
# -----------------------------------------------------------------------------
@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/api/v1/detect-image")
async def detect_image(
    file: UploadFile = File(...),
    conf_threshold: float = 0.35,
):
    # Validar tipo
    valid_types = {"image/jpeg", "image/jpg", "image/png"}
    if not file.content_type or file.content_type.lower() not in valid_types:
        raise HTTPException(
            status_code=400,
            detail=f"Formato no soportado ({file.content_type}). Use JPG o PNG."
        )

    # Leer y abrir imagen
    data = await file.read()
    try:
        img = Image.open(io.BytesIO(data))
        img.load()
        img = img.convert("RGB")
    except UnidentifiedImageError:
        raise HTTPException(status_code=400, detail="El archivo no es una imagen válida (JPEG/PNG).")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"No se pudo abrir la imagen: {e}")

    rgb = np.array(img)  # H,W,3 (RGB)

    # Inferencia (¡usar conf_thres!)
    boxes, scores, classes = DETECTOR.detect(rgb, conf_thres=conf_threshold)

    # Armar salida SIN reescalar (el detector ya devuelve coords originales XYXY)
    dets = []
    for (x1, y1, x2, y2), s, c in zip(boxes, scores, classes):
        dets.append({
            "box": [float(x1), float(y1), float(x2), float(y2)],
            "score": float(s),
            "cls": int(c),
        })

    return JSONResponse({
        "filename": file.filename,
        "count": len(dets),
        "detections": dets,
        "params": {"conf_threshold": conf_threshold},
    })

@app.post("/api/v1/detect-image-b64")
async def detect_image_b64(
    file: UploadFile = File(...),
    conf_threshold: float = 0.35
):
    if file.content_type not in {"image/jpeg", "image/png"}:
        raise HTTPException(status_code=400, detail="Formato no soportado (use JPG o PNG).")

    # Leer imagen
    try:
        data = await file.read()
        image = Image.open(io.BytesIO(data)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="No se pudo abrir la imagen")

    rgb = np.asarray(image)

    # Inferencia
    boxes, scores, classes = DETECTOR.detect(rgb, conf_thres=conf_threshold)

    detections = []
    for (x1, y1, x2, y2), s, c in zip(boxes, scores, classes):
        detections.append({
            "box": [float(x1), float(y1), float(x2), float(y2)],
            "score": float(s),
            "cls": int(c),
            "label": COCO_NAMES[int(c)] if 0 <= int(c) < len(COCO_NAMES) else str(int(c))
        })

    # Render de la imagen final con cajas -> base64
    img_bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    png_bytes = draw_boxes(img_bgr, detections)
    img_b64 = base64.b64encode(png_bytes).decode("utf-8")

    return {
        "filename": file.filename,
        "count": len(detections),
        "detections": detections,
        "image_b64": img_b64,
        "params": {"conf_threshold": conf_threshold}
    }

@app.post("/api/v1/detect-image-viz")
async def detect_image_viz(
    file: UploadFile = File(...),
    conf_threshold: float = 0.35,
):
    # Leer imagen
    data = await file.read()
    try:
        img = Image.open(io.BytesIO(data)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="No se pudo abrir la imagen")

    rgb = np.array(img)

    # Inferencia
    boxes, scores, classes = DETECTOR.detect(rgb, conf_thres=conf_threshold)

    # Armar dets sin reescalar
    dets = []
    for (x1, y1, x2, y2), s, c in zip(boxes, scores, classes):
        dets.append({
            "box": [float(x1), float(y1), float(x2), float(y2)],
            "score": float(s),
            "cls": int(c),
        })

    # Visualización PNG
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    vis = draw_detections(bgr, dets)
    vis_rgb = cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)
    buf = io.BytesIO()
    Image.fromarray(vis_rgb).save(buf, format="PNG")
    buf.seek(0)
    return StreamingResponse(buf, media_type="image/png")
