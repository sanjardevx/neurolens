# NeuroLens

Brain tumor MRI analysis for doctors: **classify** the scan, **show where the model looked**, **segment the tumor in 3D** and measure it, and let a **doctor review and approve** the result before any report leaves the system.

> ⚠️ Research and portfolio project, **not a medical device**. Results are not a diagnosis. Trained only on public datasets.

## How it works

```
doctor login → new case (anonymous ID) → upload 3D MRI → quality check
→ classifier + Grad-CAM → 3D segmentation + volume → doctor review (approve / edit / reject) → approved PDF
```

| Step | Model / tool | Output |
|---|---|---|
| Classify | EfficientNet-B3 on brain-cropped scans (timm) | glioma · meningioma · pituitary · no tumor + confidence |
| Explain | Grad-CAM | heatmap of where the model looked |
| Segment | 3D SegResNet (MONAI), BraTS 2021 | edema · tumor core · enhancing tumor masks, volume in cm³ per part, side of the brain and a rough lobe |
| Review | 3D viewer (NiiVue) | doctor approves, edits or rejects; every action logged |
| Report | template, optional LLM draft (Ollama or LLM API) | doctor-approved PDF from structured features |

## Engineering highlights

- **Leak-free data.** The public dataset has duplicate images, some in both train and test. They are found with MD5 + perceptual hashing and removed *before* training, so test scores are not inflated.
- **Honest evaluation.** Models are compared on validation only. The test split is opened only for a model already picked on val.
- **Transfer learning.** Pretrained backbones, strong augmentation, mixed precision, cosine learning rate, early stopping on macro-F1.
- **Brain crop.** Every scan is cut to the head and padded to a square, so images from different sources have the same framing.
- **Simple over complex.** Ten options were compared (two backbones, with and without crop, test-time flips, ensembles). One B3 on cropped scans tied the best two-model ensemble, so the lighter option won: one network, one pass, about half the compute (no second model, no flipped copy).
- **Explainability.** Every prediction comes with a Grad-CAM heatmap, so a wrong answer is easy to spot.
- **CPU serving without PyTorch.** Both models run in ONNX Runtime with numpy-only pre- and post-processing, inside a FastAPI service in Docker. Each step is tested to match the PyTorch / MONAI pipeline (classifier: 710 / 710 val images; 3D: masks agree at Dice ≥ 0.9997 on real scans). On a laptop CPU: 0.06 s per 2D slice, 13-20 s per 3D scan.
- **Privacy by design.** DICOM headers are never stored: a series becomes a volume of pixels + position, with a short allowlist of technical tags. Cases have anonymous codes, never names. Tested with a fake patient whose name and ID must not appear anywhere.
- **Safe reports.** The report draft is written from the numbers only (type, volumes, location), never the image. An LLM (Ollama Cloud, one `.env` setting) may reword it, but its answer is refused if it changes a number, a tumor type or drops a sentence, and the template is used instead. Only the text the doctor approves goes into the signed PDF. Examples: [sample reports](docs/sample_reports.md).
- **Doctor in the loop.** A 3D viewer (NiiVue) shows the scan with the AI mask per region; the doctor can fix the mask with a brush, then approve, edit or reject with a reason. The AI's own mask is kept next to the correction, so the difference can train the next model.
- **Feedback loop.** Doctors' corrections are exported as a versioned training set, the segmenter is fine-tuned on them, and the new version replaces the old one only if it is not worse on the held-out test split. v2 learned the fixes (0.745 → 0.841 Dice on those cases) and gained a little on test (+0.004), so it is served ([details](#feedback-loop)).
- **Traceable model versions.** Each model file carries its own version and settings; every result stores the versions that actually ran, and the worklist shows them. Releasing a model is swapping one file.
- **A worklist built for reading.** Cases waiting for review come first, largest tumors on top. Upload shows real progress, the AI job reports its own progress, and every result is labeled a draft until a doctor approves it.
- **Doctor-only, audited access.** Supabase login with two roles (doctor, admin). A doctor only reaches their own cases; every endpoint is tested with every kind of caller. Uploads, analyses and review decisions go to an audit log.

## Results

_Only measured numbers from real runs. `—` = not measured._

| Model | Metric | Target | Test result |
|---|---|---|---|
| Classifier | Accuracy | ≥ 95% | **99.4%** |
| Classifier | Macro-F1 | ≥ 95% | **99.3%** |
| 3D segmenter (v2, served) | Dice, whole tumor | ≥ 0.85 | **0.928** |
| 3D segmenter (v2, served) | Dice, tumor core | ≥ 0.75 | **0.914** |
| 3D segmenter (v2, served) | Dice, enhancing tumor | ≥ 0.70 | **0.872** |

Final classifier: EfficientNet-B3 (384 px) on brain-cropped scans, 710 clean test images, 4 errors, full precision (fp32). [Test confusion matrix](ml/results/03_final/confusion_final_test.png) · [Model card](ml/MODEL_CARD.md) (intended use, per-class scores, limitations).

**ONNX** ([report](ml/results/05_onnx/onnx_report.json), [`05_onnx_export.ipynb`](ml/notebooks/05_onnx_export.ipynb)): 43.7 MB file, 710 / 710 val predictions identical to PyTorch (max logit difference 0.00002), about 53 ms per scan on a laptop CPU (Apple M3) vs 345 ms in PyTorch.

Base segmenter (v1): SegResNet (MONAI, 4.7M parameters) on BraTS 2021, 188 test patients never used for training or tuning, Dice averaged per patient (a patient with no enhancing tumor scores 1 only if none is predicted). Trained 60 epochs in 4.8 h on one T4 GPU ([curves](ml/results/07_segresnet/curves_seg_segresnet.png)); val Dice 0.926 / 0.917 / 0.864. The enhancing score uses one rule picked on val before the single test run: predicted enhancing tumor under 50 mm³ is dropped (it is mostly a false alarm). Without it, test enhancing Dice is 0.855 ([test report](ml/results/08_seg_eval/metrics_seg_test_segresnet.json)). v1 test Dice: 0.929 whole tumor, 0.908 core, 0.866 enhancing; median per-patient Dice 0.948 / 0.958 / 0.924; median whole-tumor volume error 3.6 cm³. The served v2 is v1 fine-tuned on corrected cases ([feedback loop](#feedback-loop)).

Volumes use the scan's own voxel size (from the file header), so 3 mm slices or tilted scans are measured right. The side comes from the share of tumor right of the midline; the lobe is a position rule inside the brain box, not an atlas, and is shown to the doctor as an estimate ([`measure.py`](ml/neurolens_ml/measure.py)). Checked by eye on 6 val patients ([slices](ml/results/09_seg_onnx/onnx_seg_cases.png)): the side fits all 6, the lobe 5 of 6 (a tumor at the back of the head is called temporal; it looks occipital).

**3D ONNX** ([report](ml/results/09_seg_onnx/onnx_seg_report.json), [`09_segmenter_onnx.ipynb`](ml/notebooks/09_segmenter_onnx.ipynb)): 19.2 MB file. The serving path needs no PyTorch: NIfTI → RAS orientation → brain crop → normalize → ONNX sliding window (numpy, nibabel, ONNX Runtime; [`infer_seg.py`](ml/neurolens_ml/infer_seg.py)), tested to match the MONAI pipeline step by step. On 6 real val patients its masks agree with PyTorch (Dice ≥ 0.9997) and score the same Dice to 3 decimals. About 11 s per scan on a laptop CPU (Apple M3).

The classifier test split was opened twice: first for the earlier B0 + B3 ensemble (99.0% accuracy, 7 errors, [results](ml/results/03_final_ensemble)), then for this model, which was picked on val *before* its test run.

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

## Feedback loop

```
doctor fixes a mask → export (versioned training set) → fine-tune on Kaggle → compare on test → serve if not worse
```

No hospital has used the app yet, so the corrections are **simulated**: the 40 val patients where v1 did worst (mean Dice 0.745), with their expert BraTS masks standing in for a doctor's fix. The API's real export ([`export_feedback.py`](backend/scripts/export_feedback.py)) writes the same layout, so real corrections drop in unchanged.

Fine-tuning ([`10_feedback_finetune.ipynb`](ml/notebooks/10_feedback_finetune.ipynb)) starts from v1 with BraTS train plus the corrected cases repeated 10× per epoch (16 epochs, 1.9 h on a T4). The other 148 val patients guard against forgetting. The rule, fixed before the run: keep the epoch that learned the corrections best while staying within 0.005 mean Dice of v1 on those 148; if none beats v1, v1 stays. A first, stricter run (lower learning rate, "must beat v1 on the 148") kept v1; both runs are reported ([run 1](ml/results/10_feedback_v1kept), [run 2](ml/results/10_feedback_fb2)).

| Mean Dice | v1 | v2 |
|---|---|---|
| 40 corrected val cases (training data for v2: shows learning) | 0.745 | **0.841** |
| 148 other val cases (guard) | 0.943 | 0.943 |
| **188 test patients** (never seen by either; v1 as served, with its 50 mm³ ET rule) | 0.901 | **0.905** |
| 37 test patients with a small enhancing tumor (< 5 cm³) | 0.785 | 0.792 |

Test ([`11_compare_versions.ipynb`](ml/notebooks/11_compare_versions.ipynb)) was opened once for both models, each with its small-ET rule tuned on the 148 guard cases (both came out 0). With that preset rule v2 gains +0.007 (95% bootstrap interval +0.002 to +0.014) and the decision rule ("keep v2 if at most 0.005 below v1") kept v2. Checked afterwards: much of that gain is two patients without enhancing tumor where v1 drew a 3–7 voxel speck, which v1's own served 50 mm³ rule already removes. **Against v1 as served, the gain is +0.004 (interval +0.001 to +0.007)**: 17 patients improve, 159 stay within 0.01, 12 drop (the worst by 0.036). On small enhancing tumors v2 is not clearly better (+0.008, interval −0.002 to +0.021). So v2 learned the corrections, did not forget, and is a small, real step up overall. The segmentation test split had been opened once before, for v1 alone (notebook 08); this was its second and last use. [Per-patient chart](ml/results/11_compare_versions/compare_test_seg.png).

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
| [BraTS 2021, Task 1](https://www.kaggle.com/datasets/dschettler8845/brats-2021-task1) | 3D segmentation: 4 MRI types (T1, T1ce, T2, FLAIR) + expert masks, split by patient |

BraTS 2021 is shared under [CC BY 4.0](https://doi.org/10.7937/jc8x-9874). Please cite Baid et al. 2021 ([arXiv:2107.02314](https://arxiv.org/abs/2107.02314)), Menze et al. 2015 and Bakas et al. 2017. It is checked in [`06_brats_data.ipynb`](ml/notebooks/06_brats_data.ipynb) ([report](ml/data/brats_report.json)):

| BraTS check | Value |
|---|---|
| Patients with all 4 MRI types + mask | 1,251 |
| Volume size and voxel | 240 × 240 × 155, 1 mm (all cases) |
| Mask labels | 0 background, 1 necrotic core, 2 edema, 4 enhancing (no others) |
| Whole tumor volume, median (range) | 89 cm³ (3 – 362) |
| Cases with no enhancing part | 33 |
| Split by patient (train / val / test) | 875 / 188 / 188, stratified by tumor size ([`brats_splits.csv`](ml/data/brats_splits.csv)) |
| Masks line up with the scans | yes: 100% of tumor voxels inside the brain in the [checked patients](ml/results/06_brats/brats_slices.png), also after the [training transforms](ml/results/06_brats/brats_patch.png) |

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
| ML | PyTorch, timm, MONAI, Albumentations |
| Serving | ONNX Runtime, FastAPI, Docker |
| Frontend | Next.js, Tailwind, shadcn/ui, NiiVue |
| Data & reports | Supabase, Ollama / LLM API |
| CI / Deploy | GitHub Actions, Hugging Face Spaces, Vercel |

## Project layout

```
ml/neurolens_ml/  Python package: data, models, train / evaluate / export CLIs
ml/configs/     one YAML file per training run
ml/notebooks/   training notebooks (run on Kaggle GPU)
ml/data/        data splits and data reports (no images in git)
ml/results/     metrics, confusion matrices, Grad-CAM figures
ml/MODEL_CARD.md  what the classifier is for, how it was tested, its limits
backend/        FastAPI + ONNX Runtime: login, cases, upload, AI jobs, review, audit log (see backend/README.md)
frontend/       Next.js dashboard for doctors: login, worklist, upload, 3D review, audit log (see frontend/README.md)
```

## Run it

```bash
pip install -e "ml[export,dev,seg]"
pytest ml                       # leak-free splits, crop, model outputs, tiny 2D and 3D training runs, BraTS loading and 3D transforms
python -m neurolens_ml.train    --config ml/configs/cls_b3.yaml --data <dataset folder>
python -m neurolens_ml.evaluate --config ml/configs/cls_b3.yaml --checkpoint outputs/classifier_strong_crop.pt
python -m neurolens_ml.export   --config ml/configs/cls_b3.yaml --checkpoint outputs/classifier_strong_crop.pt
python -m neurolens_ml.prep_brats --data <BraTS folder>   # check the 3D data: cases, labels, tumor volumes
python -m neurolens_ml.split_brats                        # patient split from brats_cases.csv, no scans needed
python -m neurolens_ml.plot_brats  --data <BraTS folder>   # slices with the expert mask on top
python -m neurolens_ml.train_seg  --config ml/configs/seg_segresnet.yaml --data <BraTS folder>
python -m neurolens_ml.finetune_seg --config ml/configs/seg_finetune.yaml --data <BraTS folder>  # feedback loop
python -m neurolens_ml.compare_seg --old-config ml/configs/seg_segresnet.yaml --new-config ml/configs/seg_finetune.yaml
```

`evaluate` uses val by default; `--split test` is for the final model only. On Kaggle, [`02_classifier.ipynb`](ml/notebooks/02_classifier.ipynb) clones the repo and runs the same `train` command on a GPU.

API (needs `.env` with the Supabase settings and the two `.onnx` files in `ml/models/`):

```bash
pip install -e "backend[dev]" && pip install --no-deps -e ml
pytest backend                  # auth, roles on every endpoint, uploads and privacy, AI jobs, reviews
docker compose up --build       # API on http://localhost:8000 (docs at /docs)
python backend/scripts/smoke_flow.py --url http://localhost:8000 --doctor-token <jwt> --admin-token <jwt>
python backend/scripts/export_feedback.py --out feedback --zip   # doctors' corrections → training set for finetune_seg
```

Dashboard (needs `frontend/.env.local` with the Supabase URL and publishable key, and the API running):

```bash
npm --prefix frontend install
npm --prefix frontend run dev   # http://localhost:3000
```

## License

[MIT](LICENSE). Datasets keep their own licenses.
