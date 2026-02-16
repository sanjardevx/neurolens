# NeuroLens API

FastAPI service for doctors: login (Supabase Auth), cases, scans, AI results, review and an audit log.
Doctor-only: patients have no accounts and never see AI output.

```bash
pip install -e "backend[dev]"
pip install --no-deps -e ml          # inference code only (numpy, OpenCV, nibabel, ONNX Runtime); no PyTorch
pytest backend
uvicorn app.main:app --app-dir backend --reload   # from the repo root; reads .env there
```

`.env` needs `SUPABASE_URL` and `DATABASE_URL`. `CORS_ORIGINS` lists the dashboard origins allowed to call the API
(default `http://localhost:3000`).

Models: `MODEL_DIR` must hold `classifier.onnx` and `segmenter3d.onnx` (made by `neurolens_ml.export` and
`neurolens_ml.export_seg`; not in git). Tests use tiny stand-in ONNX models with the same inputs and outputs.

Model versions: each ONNX file carries its version (and the segmenter its small-ET limit) inside it
(`neurolens_ml.onnx_meta`). A result stores the versions of the files that actually ran, so releasing a model is
swapping the file and restarting; old results keep the version that made them. A file without a version is refused.

## Deploy

The live API runs on Google Cloud Run (2 vCPU, 4 GiB, scales to zero). After CI passes on `master`,
`.github/workflows/deploy.yml` builds `backend/Dockerfile` (which downloads the two ONNX files from the public model
repo in `MODEL_REPO`), pushes the image to Artifact Registry and deploys it. GitHub logs in to Google Cloud without a
stored key (Workload Identity Federation, limited to this repo).

| Setting | Where | Value |
|---|---|---|
| `GCP_PROJECT`, `GCP_REGION` | GitHub variables | Google Cloud project id; region (same as the Supabase database) |
| `GCP_WIF_PROVIDER`, `GCP_DEPLOY_SA`, `GCP_RUN_SA` | GitHub variables | the GitHub identity provider; the deploy and runtime service accounts |
| `MODEL_REPO` | GitHub variable | Hugging Face model repo with `classifier.onnx` + `segmenter3d.onnx` |
| `SUPABASE_URL`, `CORS_ORIGINS` | GitHub variables | project URL; the dashboard's URL |
| `DATABASE_URL`, `SUPABASE_SECRET_KEY` | Google Secret Manager | `neurolens-database-url`, `neurolens-supabase-secret-key` |

The live demo writes template reports only (no LLM key on the server). A Cloud Run instance loses its disk when it
stops, so every scan and result folder is mirrored to the private Supabase bucket and downloaded back when needed
(`app/files.py`). Cloud Run accepts at most 32 MB per request: upload `.nii.gz` files, not raw `.nii`.

## Roles

Only two roles: `doctor` and `admin`. The role is read from the login token's `app_metadata.role`,
which users cannot change themselves. Set it once per account in the Supabase SQL editor:

```sql
update auth.users
set raw_app_meta_data = raw_app_meta_data || '{"role": "doctor"}'::jsonb   -- or "admin"
where email = 'doctor@example.com';
```

The new role is in the token after the next login.

| | doctor | admin |
|---|---|---|
| Create a case, upload, review | own cases only | no |
| See cases, scans, results, reviews | own only (others' return 404) | all |
| Read the audit log | no | yes |

Every endpoint is tested with every kind of caller (case owner, other doctor, admin, no login, no role) in
[`tests/test_access.py`](tests/test_access.py).

## Uploads

`POST /cases/{id}/scans` (the case's doctor only) takes either:
- the 4 MRI types for 3D segmentation: `flair`, `t1`, `t1ce`, `t2`, each a `.nii` / `.nii.gz` or a `.zip` of one DICOM series, or
- one 2D slice as `image` (JPG / PNG) for the classifier.

Checks before anything is saved (a failed check returns 422 with the reason, and only the reason is logged):
MR modality and brain body part (DICOM), one series per zip, all 4 types present, same size and position
(co-registered), 32-512 voxels and 0.3-7 mm per axis, skull-stripped (the 3D model was trained on BraTS),
and for images: grayscale, dark background, 128-4096 px. These are basic checks, not a guarantee; the doctor reviews every case.

## AI analysis

A successful upload starts the analysis in the background; the case goes `processing` → `ready` (or `failed` with the reason).
`GET /cases/{id}/results` shows progress (0-100) and the outcome, with the model version on every result.

| Upload | Output | CPU time (Apple M3) |
|---|---|---|
| 2D slice | tumor type, confidence, all class probabilities, Grad-CAM image | ~0.06 s |
| 4 MRI types | 3D mask (BraTS labels 1 / 2 / 4, `GET .../results/{rid}/mask`), volume per part in cm³, side, rough lobe; tumor type + Grad-CAM from the T1ce slice with the most tumor | ~14 s |

For 3D uploads the tumor type comes from the 2D classifier on one slice; the result says so (`info.classifier_input`).

## Doctor review

`POST /cases/{id}/reviews` with the result being reviewed:

| Decision | Needs | Use |
|---|---|---|
| `approved` | nothing else | the AI result is right |
| `edited` | `report_text` and / or `corrected_type` | right with changes |
| `rejected` | `reason`: `wrong_type`, `bad_mask`, `artifact` or `other` (+ `note`) | not usable |

Only a finished analysis can be reviewed (409 otherwise). A new review replaces the previous one; the history and
every decision stay in the audit log. Edits and rejections are the training signal for the Week 9 retraining.

## Data safety

- Tables: `cases`, `scans`, `results`, `reviews`, `audit_log`. A case has an anonymous code (`NL-7KQ2XD`), never a name.
- On Supabase, row-level security is switched on with no policies, so the public REST API sees nothing;
  only this API (database owner connection) reads and writes.
- **DICOM headers are never stored.** A series becomes a NIfTI volume (pixels + position in space, RAS orientation); only technical tags are kept (modality, body part, field strength, slice thickness, TR, TE): no names, IDs, dates, hospital or free text. NIfTI files get a fresh header; images are re-encoded as PNG without EXIF. Text burned into the pixels is not removed.
- Creating and opening a case, uploads, rejected uploads, finished / failed analyses and every review decision are written to `audit_log`.

## Report drafts

`GET /cases/{id}/results/{rid}/report` returns the current draft: findings, impression and notes written by a
template from the numbers only (tumor type, confidence, volumes, side, rough lobe), with the doctor's type and
mask corrections taking priority. `POST .../report/draft` asks an LLM to reword it, through one OpenAI-compatible
client set in `.env` (`REPORT_PROVIDER=llm`, `LLM_BASE_URL`, `LLM_API_KEY`, `LLM_MODEL`; Ollama Cloud or local
Ollama). The LLM never sees the image. Its answer is refused, and the template returned with the reason, if it
adds a number, names a tumor type the draft didn't, drops a sentence or isn't valid JSON. A saved LLM draft is
only reused while the numbers it was written from stay the same.

## Feedback export

`python backend/scripts/export_feedback.py --out feedback [--tag fb-v1] [--zip]` turns reviewed 3D cases into a
versioned training set: **train** (the doctor fixed the mask; the corrected mask is the label), **relabel** (rejected
for a bad mask but not fixed), **excluded** (artifact or type-only rejection) and **approved** (counted, not exported).
Files use the BraTS layout on one RAS grid, so `neurolens_ml` reads them directly. `manifest.csv` lists every case with
the AI vs doctor whole-tumor Dice; `summary.json` has the counts and model versions. No names, emails or user ids.

