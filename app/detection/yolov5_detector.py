# app/detection/yolov5_detector.py
from __future__ import annotations
import numpy as np
import onnxruntime as ort
import cv2
from typing import List, Tuple
from .base import BaseDetector

# NMS sencillo
def nms(boxes: np.ndarray, scores: np.ndarray, iou_thres: float = 0.5) -> List[int]:
    if boxes.size == 0:
        return []
    x1, y1, x2, y2 = boxes.T
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(int(i))
        if order.size == 1:
            break
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-9)
        inds = np.where(iou <= iou_thres)[0]
        order = order[inds + 1]
    return keep

# letterbox como en YOLOv5 (resize con padding para mantener aspecto)
def letterbox(img: np.ndarray, new_size: int = 640, color=(114, 114, 114)) -> Tuple[np.ndarray, float, Tuple[int,int]]:
    h, w = img.shape[:2]
    scale = min(new_size / h, new_size / w)
    nh, nw = int(round(h * scale)), int(round(w * scale))
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    top = (new_size - nh) // 2
    bottom = new_size - nh - top
    left = (new_size - nw) // 2
    right = new_size - nw - left
    out = cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return out, scale, (left, top)

class OnnxYoloV5Detector(BaseDetector):
    """
    Detector YOLOv5 ONNX (COCO 80 clases).
    Salida esperada del modelo: (1, 25200, 85) -> [x,y,w,h, obj, 80 clases]
    """

    def __init__(self, model_path: str = "models/yolov5s.onnx", img_size: int = 640, providers=None):
        self.model_path = model_path
        self.img_size = img_size
        self.providers = providers or ["CPUExecutionProvider"]
        self.session = None
        self.inp_name = None
        self.out_name = None
        self.load_model()

    def load_model(self) -> None:
        self.session = ort.InferenceSession(self.model_path, providers=self.providers)
        self.inp_name = self.session.get_inputs()[0].name
        self.out_name = self.session.get_outputs()[0].name

    def detect(self, image: np.ndarray, conf_thres: float = 0.25, iou_thres: float = 0.50):
        """
        image: RGB np.uint8 (H,W,3)
        return: (boxes_xyxy, scores, classes) en coordenadas de la imagen original
        """
        assert image.ndim == 3 and image.shape[2] == 3, "La imagen debe ser RGB (H,W,3)"
        H, W = image.shape[:2]

        # 1) Preprocesado (letterbox 640, normalizar, BCHW)
        lb, scale, (padx, pady) = letterbox(image, self.img_size)
        inp = lb.astype(np.float32) / 255.0
        inp = inp.transpose(2, 0, 1)[None, ...]  # 1x3x640x640

        # 2) Inferencia
        out = self.session.run([self.out_name], {self.inp_name: inp})[0]  # (1, 25200, 85)
        preds = out[0]  # (25200, 85)

        # 3) Decodificar (cxcywh -> xyxy) y calcular score = obj * cls_conf
        cx, cy, w, h = preds[:, 0], preds[:, 1], preds[:, 2], preds[:, 3]
        obj = preds[:, 4]
        cls_scores = preds[:, 5:]  # (25200, 80)
        cls_ids = cls_scores.argmax(axis=1)
        cls_conf = cls_scores.max(axis=1)
        scores = obj * cls_conf

        # filtrar por confianza
        mask = scores >= conf_thres
        if not np.any(mask):
            return np.empty((0,4), dtype=np.float32), np.array([], dtype=np.float32), np.array([], dtype=np.int32)

        cx, cy, w, h = cx[mask], cy[mask], w[mask], h[mask]
        scores = scores[mask]
        cls_ids = cls_ids[mask]

        x1 = cx - w / 2
        y1 = cy - h / 2
        x2 = cx + w / 2
        y2 = cy + h / 2
        boxes = np.stack([x1, y1, x2, y2], axis=1)

        # 4) NMS por clase
        final_boxes, final_scores, final_classes = [], [], []
        for c in np.unique(cls_ids):
            idxs = np.where(cls_ids == c)[0]
            keep = nms(boxes[idxs], scores[idxs], iou_thres=iou_thres)
            if keep:
                final_boxes.append(boxes[idxs][keep])
                final_scores.append(scores[idxs][keep])
                final_classes.append(np.full(len(keep), c, dtype=np.int32))

        if not final_boxes:
            return np.empty((0,4), dtype=np.float32), np.array([], dtype=np.float32), np.array([], dtype=np.int32)

        boxes = np.concatenate(final_boxes, axis=0)
        scores = np.concatenate(final_scores, axis=0)
        classes = np.concatenate(final_classes, axis=0)

        # 5) Reescalar de 640 padded a tamaño original
        # primero quitar el padding del letterbox
        boxes[:, [0, 2]] -= padx
        boxes[:, [1, 3]] -= pady
        boxes /= scale

        # clamp a la imagen original
        boxes[:, 0::2] = np.clip(boxes[:, 0::2], 0, W - 1)
        boxes[:, 1::2] = np.clip(boxes[:, 1::2], 0, H - 1)

        return boxes.astype(np.float32), scores.astype(np.float32), classes.astype(np.int32)
