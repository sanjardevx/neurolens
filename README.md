# NeuroLens

Brain tumor MRI analysis in one dashboard: **classify** the scan, **show where the model looked**, **outline the tumor**, and write a **plain-language report**.

> ⚠️ Research and portfolio project, **not a medical device**. Results are not a diagnosis. Trained only on public datasets.

## How it works

```
MRI upload → quality check → classifier → Grad-CAM → segmentation → features → report → dashboard
```

| Step | Model / tool | Output |
|---|---|---|
| Classify | EfficientNet-B3 on brain-cropped scans (timm) | glioma · meningioma · pituitary · no tumor + confidence |
| Explain | Grad-CAM | heatmap of where the model looked |
| Segment | U-Net (segmentation_models_pytorch) | tumor mask + area |
| Report | Ollama or LLM API, template fallback | short summary from structured features |

## Engineering highlights

- **Leak-free data.** The public dataset has duplicate images, some in both train and test. They are found with MD5 + perceptual hashing and removed *before* training, so test scores are not inflated.
- **Honest evaluation.** Models are compared on validation only. The test split is opened only for a model already picked on val.
- **Transfer learning.** Pretrained backbones, strong augmentation, mixed precision, cosine learning rate, early stopping on macro-F1.
- **Brain crop.** Every scan is cut to the head and padded to a square, so images from different sources have the same framing.
- **Simple over complex.** Ten options were compared (two backbones, with and without crop, test-time flips, ensembles). One B3 on cropped scans tied the best two-model ensemble, so the lighter option won: one network, one pass, about half the compute (no second model, no flipped copy).
- **Explainability.** Every prediction comes with a Grad-CAM heatmap, so a wrong answer is easy to spot.

## Results

_Only measured numbers from real runs. `—` = not measured._

| Model | Metric | Target | Test result |
|---|---|---|---|
| Classifier | Accuracy | ≥ 95% | **99.4%** |
| Classifier | Macro-F1 | ≥ 95% | **99.3%** |
| Segmenter | Dice | ≥ 0.85 | — |

Final classifier: EfficientNet-B3 (384 px) on brain-cropped scans, 710 clean test images, 4 errors, full precision (fp32). [Test confusion matrix](ml/results/03_final/confusion_final_test.png).

The test split was opened twice: first for the earlier B0 + B3 ensemble (99.0% accuracy, 7 errors, [results](ml/results/03_final_ensemble)), then for this model, which was picked on val *before* its test run.

**How the final model was picked** (validation split, 710 images, [`03_final_classifier.ipynb`](ml/notebooks/03_final_classifier.ipynb)):

| Option | Val accuracy | Val macro-F1 | Errors |
|---|---|---|---|
| **B3 + brain crop** | **98.5%** | **98.5%** | **11** |
| B0 + B3 ensemble + TTA | 98.5% | 98.4% | 11 |
| B3 + brain crop + TTA | 98.0% | 98.0% | 14 |
| B0 + B3 crop ensemble + TTA | 98.0% | 97.9% | 14 |
| B0 + TTA | 97.8% | 97.6% | 16 |
| B3 + TTA | 97.6% | 97.6% | 17 |
| B0 + brain crop | 97.6% | 97.6% | 17 |
| B0 + brain crop + TTA | 97.3% | 97.1% | 19 |
| EfficientNet-B0, 224 px | 97.6% | 97.3% | 17 |
| EfficientNet-B3, 384 px, strong augmentation | 97.5% | 97.4% | 18 |

TTA = the prediction is averaged with a left-right flipped copy. Training runs: [`02_crop`](ml/results/02_crop) (final), [`02_baseline_rerun`](ml/results/02_baseline_rerun), [`02_strong`](ml/results/02_strong), [`02_baseline`](ml/results/02_baseline).

## Explainability

![Grad-CAM examples](ml/results/04_gradcam_final/gradcam_examples_final.png)

The heatmap is the classifier weights × the last feature maps. For this model head that equals Grad-CAM (checked numerically in [`04_gradcam.ipynb`](ml/notebooks/04_gradcam.ipynb)), so it can be computed from an ONNX model's outputs, without PyTorch.

**What the heatmaps tell us (limitation):**
- **Meningioma and many glioma maps sit on the tumor.** Pituitary maps often light up the ventricles instead of the gland, and no-tumor maps are weak and spread out.
- **Almost every map has a small hot spot in the top-left corner,** even on black padding. It looks like a network edge effect, not the tumor.
- **The brain crop did not remove the shortcut.** It was added because this public dataset mixes several sources, roughly one per class, so a model can learn the look of each source. It kept the score, but the heat outside the head did not drop. Measured against a flat heatmap (= 1.0): the two-model ensemble went from 0.51 without crop to 0.50 with it, and the final single model scores 0.64, since one model's map is not smoothed by averaging.
- **The 99% is a score on this dataset, not a promise for real hospital scans.**

More: [per class](ml/results/04_gradcam_final/gradcam_per_class_final.png), [on the mistakes](ml/results/04_gradcam_final/gradcam_errors_final.png), [earlier ensemble without crop](ml/results/04_gradcam), [ensemble with crop](ml/results/04_gradcam_crop).

## Data

| Dataset | Used for |
|---|---|
| [Brain Tumor MRI Dataset](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset) | classification, 4 classes |
| [LGG MRI Segmentation](https://www.kaggle.com/datasets/mateuszbuda/lgg-mri-segmentation) | segmentation masks |

Cleaning is done in [`01_data_prep.ipynb`](ml/notebooks/01_data_prep.ipynb):

| Data cleaning | Value |
|---|---|
| Images in the original dataset | 7,200 |
| Exact duplicate files | 187 |
| Unique scans after grouping near-copies | 4,755 |
| Scans found in both the original train and test folders | 520 |
| Groups dropped for conflicting labels | 25 |
| Images kept (train / val / test) | 4,730 (3,310 / 710 / 710) |

Many images are the same scan saved again with other contrast or sharpening. In the `notumor` class only 573 of 1,800 images are unique. Because the original test folder is mostly copies of training images, the clean scans are split again into 70 / 15 / 15, stratified by class.

## Tech stack

| Area | Tools |
|---|---|
| ML | PyTorch, timm, segmentation_models_pytorch, Albumentations |
| Serving | ONNX Runtime, FastAPI, Docker |
| Frontend | Next.js, Tailwind, shadcn/ui, Recharts |
| Data & reports | Supabase, Ollama / LLM API |
| CI / Deploy | GitHub Actions, Hugging Face Spaces, Vercel |

## Project layout

```
ml/notebooks/   training notebooks (run on Kaggle GPU)
ml/data/        splits.csv + data_report.json (no images in git)
ml/results/     metrics, confusion matrices, Grad-CAM figures
```

## License

[MIT](LICENSE). Datasets keep their own licenses.
