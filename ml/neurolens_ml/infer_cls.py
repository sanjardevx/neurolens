"""2D tumor-type classification without PyTorch: brain crop → resize → normalize → ONNX → class, confidence, Grad-CAM.

Used by training code (preprocess) and by the API, which installs only numpy, opencv and onnxruntime.
"""
from collections.abc import Callable

import cv2
import numpy as np

from .crop import crop_brain

CLASSES = ["glioma", "meningioma", "notumor", "pituitary"]
MEAN = (0.485, 0.456, 0.406)  # ImageNet stats, to match the pretrained backbones
STD = (0.229, 0.224, 0.225)
IMG_SIZE = 384  # final model: configs/cls_b3.yaml


def preprocess(img: np.ndarray, size: int, crop: bool = True) -> np.ndarray:
    """Numpy-only version of the eval transform (for ONNX serving): RGB uint8 -> float32 [1, 3, size, size]."""
    if crop:
        img = crop_brain(img)
    img = cv2.resize(img, (size, size), interpolation=cv2.INTER_LINEAR)
    x = (img.astype(np.float32) / 255.0 - np.array(MEAN, np.float32)) / np.array(STD, np.float32)
    return np.ascontiguousarray(x.transpose(2, 0, 1)[None])


def softmax(logits: np.ndarray) -> np.ndarray:
    e = np.exp(logits - logits.max())
    return e / e.sum()


def gradcam_overlay(img: np.ndarray, cam: np.ndarray, alpha: float = 0.4) -> np.ndarray:
    """Heatmap of one class (weights x feature maps = Grad-CAM for this head) over the RGB input."""
    cam = np.maximum(cam, 0)
    cam = cam / cam.max() if cam.max() > 0 else cam
    cam = cv2.resize(cam.astype(np.float32), img.shape[1::-1], interpolation=cv2.INTER_LINEAR)
    heat = cv2.cvtColor(cv2.applyColorMap((cam * 255).astype(np.uint8), cv2.COLORMAP_JET), cv2.COLOR_BGR2RGB)
    return ((1 - alpha) * img + alpha * heat).astype(np.uint8)


def classify(img: np.ndarray, run: Callable[[np.ndarray], tuple[np.ndarray, np.ndarray]],
             size: int = IMG_SIZE) -> dict:
    """RGB slice → tumor type, confidence, all class probabilities and a Grad-CAM image of the predicted class."""
    cropped = cv2.resize(crop_brain(img), (size, size), interpolation=cv2.INTER_LINEAR)
    logits, cams = run(preprocess(cropped, size, crop=False))
    probs = softmax(logits[0])
    k = int(probs.argmax())
    return {"tumor_type": CLASSES[k], "confidence": float(probs[k]),
            "probabilities": {c: round(float(p), 4) for c, p in zip(CLASSES, probs)},
            "gradcam": gradcam_overlay(cropped, cams[0, k])}


def onnx_classifier(path: str, threads: int | None = None) -> Callable[[np.ndarray], tuple[np.ndarray, np.ndarray]]:
    import onnxruntime as ort

    opts = ort.SessionOptions()
    if threads:
        opts.intra_op_num_threads = threads
    session = ort.InferenceSession(str(path), opts, providers=["CPUExecutionProvider"])
    return lambda x: tuple(session.run(["logits", "cams"], {"image": x.astype(np.float32)}))
