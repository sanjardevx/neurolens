---
license: mit
tags: [medical-imaging, brain-tumor, mri, segmentation, onnx]
---

# NeuroLens models (ONNX)

The two models served by [NeuroLens](https://github.com/sanjardevx/neurolens). Research and portfolio project,
**not a medical device**; not for diagnosis.

| File | Version (inside the file) | What it does | Test result |
|---|---|---|---|
| `segmenter3d.onnx` | `seg-segresnet-v2` | 3D SegResNet: edema / tumor core / enhancing tumor from 4 co-registered, skull-stripped MRI types (BraTS style), 128³ sliding window | Dice 0.928 / 0.914 / 0.872 (whole / core / enhancing), 188 BraTS 2021 test patients |
| `classifier.onnx` | `cls-b3-crop-v1` | EfficientNet-B3 on one 2D slice: glioma / meningioma / pituitary / no tumor, plus class activation maps | 99.4% accuracy, 710 clean test images |

Each file stores its version and settings as ONNX metadata (`neurolens_ml.onnx_meta`); the segmenter also stores
its small-enhancing-tumor limit. Trained on public data only: BraTS 2021 (CC BY 4.0) and the Brain Tumor MRI Dataset.
How they were trained, tested and compared: the [project README](https://github.com/sanjardevx/neurolens#results).
