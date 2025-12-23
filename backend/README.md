# NeuroLens API

FastAPI service for doctors: login (Supabase Auth), cases, scans, AI results, review and an audit log.
Doctor-only: patients have no accounts and never see AI output.

```bash
pip install -e "backend[dev]"
pytest backend
cd backend && uvicorn app.main:app --reload   # needs SUPABASE_URL and DATABASE_URL in .env
```

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
| Create a case | yes | no |
| See cases | own only (others' return 404) | all |
| Read the audit log | no | yes |

## Uploads

`POST /cases/{id}/scans` (the case's doctor only) takes either:
- the 4 MRI types for 3D segmentation: `flair`, `t1`, `t1ce`, `t2`, each a `.nii` / `.nii.gz` or a `.zip` of one DICOM series, or
- one 2D slice as `image` (JPG / PNG) for the classifier.

Checks before anything is saved (a failed check returns 422 with the reason, and only the reason is logged):
MR modality and brain body part (DICOM), one series per zip, all 4 types present, same size and position
(co-registered), 32-512 voxels and 0.3-7 mm per axis, skull-stripped (the 3D model was trained on BraTS),
and for images: grayscale, dark background, 128-4096 px. These are basic checks, not a guarantee; the doctor reviews every case.

## Data safety

- Tables: `cases`, `scans`, `results`, `reviews`, `audit_log`. A case has an anonymous code (`NL-7KQ2XD`), never a name.
- On Supabase, row-level security is switched on with no policies, so the public REST API sees nothing;
  only this API (database owner connection) reads and writes.
- **DICOM headers are never stored.** A series becomes a NIfTI volume (pixels + position in space, RAS orientation); only technical tags are kept (modality, body part, field strength, slice thickness, TR, TE): no names, IDs, dates, hospital or free text. NIfTI files get a fresh header; images are re-encoded as PNG without EXIF. Text burned into the pixels is not removed.
- Creating and opening a case, uploads and rejected uploads are written to `audit_log`.
