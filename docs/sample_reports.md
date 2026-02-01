# Sample report drafts

Made-up measurement sets (no patients, no images) showing the template and the LLM wording side by side.
LLM: gpt-oss:120b. The LLM sees only the numbers and the template draft; its answer is refused (template used) if it changes a number, a tumor type or the sentence count.

## Glioma, right parietal

**Template**

```
Findings:
- A tumor is outlined in the right hemisphere, centered in the parietal region (estimate).
- Whole tumor volume 89.3 cm³ (from the 3D segmentation): enhancing tumor 21.6 cm³, necrotic / non-enhancing core 12.6 cm³, surrounding edema 55.1 cm³.
- The 2D classifier suggests glioma (94%, not calibrated).

Impression: Findings suggest a glioma of about 89 cm³, in the right hemisphere, centered in the parietal region (estimate).

Notes:
- The region is a rough position estimate, not an atlas label.

AI-assisted draft from a research prototype, not a medical device. The reviewing doctor makes the diagnosis.
```

**LLM** (gpt-oss:120b, 2.0 s, passed the checks)

```
Findings:
- A tumor is outlined in the right hemisphere, centered in the parietal region (estimate).
- Whole tumor volume 89.3 cm³ (from the 3D segmentation): enhancing tumor 21.6 cm³, necrotic / non-enhancing core 12.6 cm³, surrounding edema 55.1 cm³.
- The 2D classifier suggests glioma (94%, not calibrated).

Impression: Findings suggest a glioma measuring approximately 89.3 cm³ in the right hemisphere, centered in the parietal region (estimate).

Notes:
- The region is a rough position estimate, not an atlas label.

AI-assisted draft from a research prototype, not a medical device. The reviewing doctor makes the diagnosis.
```

## Large glioma crossing the midline

**Template**

```
Findings:
- A tumor is outlined crossing the midline (both hemispheres), centered in the frontal region (estimate).
- Whole tumor volume 146.2 cm³ (from the 3D segmentation): enhancing tumor 38.4 cm³, necrotic / non-enhancing core 29.9 cm³, surrounding edema 77.9 cm³.
- The 2D classifier suggests glioma (88%, not calibrated).

Impression: Findings suggest a glioma of about 146 cm³, crossing the midline (both hemispheres), centered in the frontal region (estimate).

Notes:
- The region is a rough position estimate, not an atlas label.

AI-assisted draft from a research prototype, not a medical device. The reviewing doctor makes the diagnosis.
```

**LLM** (gpt-oss:120b, 1.9 s, passed the checks)

```
Findings:
- A tumor is delineated crossing the midline, involving both hemispheres, centered in the frontal region (estimate).
- Whole tumor volume is 146.2 cm³ (3D segmentation), with enhancing component 38.4 cm³, necrotic/non‑enhancing core 29.9 cm³, and surrounding edema 77.9 cm³.
- The 2D classifier suggests glioma (88%, not calibrated).

Impression: Findings are consistent with a glioma measuring approximately 146.2 cm³, crossing the midline (both hemispheres), centered in the frontal region (estimate).

Notes:
- The region is a rough position estimate, not an atlas label.

AI-assisted draft from a research prototype, not a medical device. The reviewing doctor makes the diagnosis.
```

## Small lesion, models disagree

**Template**

```
Findings:
- A tumor is outlined in the left hemisphere, centered in the temporal region (estimate).
- Whole tumor volume 3.2 cm³ (from the 3D segmentation): enhancing tumor 0.6 cm³, necrotic / non-enhancing core 0.5 cm³, surrounding edema 2.1 cm³.
- The 2D classifier suggests no tumor (71%, not calibrated).

Impression: Findings suggest a tumor of about 3 cm³, in the left hemisphere, centered in the temporal region (estimate).

Notes:
- The classifier says no tumor, but the 3D segmentation outlined one: check the scan closely.
- Small lesion (under 6 cm³): the 3D model is least reliable at this size.
- The region is a rough position estimate, not an atlas label.

AI-assisted draft from a research prototype, not a medical device. The reviewing doctor makes the diagnosis.
```

**LLM** (gpt-oss:120b, 2.6 s, passed the checks)

```
Findings:
- A tumor is delineated in the left hemisphere, centered in the temporal region (estimate).
- 3D segmentation shows a whole tumor volume of 3.2 cm³, comprising an enhancing component of 0.6 cm³, a necrotic/non‑enhancing core of 0.5 cm³, and surrounding edema measuring 2.1 cm³.
- The 2D classifier indicates no tumor (71% confidence, not calibrated).

Impression: A left temporal tumor measuring approximately 3 cm³ (estimate).

Notes:
- The classifier says no tumor, but the 3D segmentation outlined one: check the scan closely.
- Small lesion (under 6 cm³): the 3D model is least reliable at this size.
- The region is a rough position estimate, not an atlas label.

AI-assisted draft from a research prototype, not a medical device. The reviewing doctor makes the diagnosis.
```

## No tumor

**Template**

```
Findings:
- The 3D segmentation outlined no tumor.
- The 2D classifier suggests no tumor (97%, not calibrated).

Impression: No tumor outlined by the AI.

AI-assisted draft from a research prototype, not a medical device. The reviewing doctor makes the diagnosis.
```

**LLM** (gpt-oss:120b, 1.2 s, passed the checks)

```
Findings:
- 3D segmentation shows no tumor.
- 2D classifier indicates no tumor (confidence 97%, not calibrated).

Impression: No tumor identified by AI analysis.

AI-assisted draft from a research prototype, not a medical device. The reviewing doctor makes the diagnosis.
```

## 2D slice, pituitary

**Template**

```
Findings:
- Single 2D slice: tumor type only, no volume or location measured.
- The 2D classifier suggests pituitary tumor (81%, not calibrated).

Impression: Findings suggest a pituitary tumor on the uploaded slice.

AI-assisted draft from a research prototype, not a medical device. The reviewing doctor makes the diagnosis.
```

**LLM** (gpt-oss:120b, 1.5 s, passed the checks)

```
Findings:
- Single 2D slice shows tumor type only; volume and location not measured.
- 2D classifier suggests pituitary tumor (81%, not calibrated).

Impression: Findings are consistent with a pituitary tumor on the examined slice.

AI-assisted draft from a research prototype, not a medical device. The reviewing doctor makes the diagnosis.
```

## Doctor changed the type and fixed the mask

**Template**

```
Findings:
- A tumor is outlined in the left hemisphere, centered in the frontal region (estimate).
- Whole tumor volume 24.8 cm³ (from the doctor-corrected mask): enhancing tumor 9.1 cm³, necrotic / non-enhancing core 1.2 cm³, surrounding edema 14.5 cm³.
- The 2D classifier suggests glioma (62%, not calibrated).
- Tumor type set by the reviewing doctor: meningioma.

Impression: Findings suggest a meningioma of about 25 cm³, in the left hemisphere, centered in the frontal region (estimate).

Notes:
- The region is a rough position estimate, not an atlas label.

AI-assisted draft from a research prototype, not a medical device. The reviewing doctor makes the diagnosis.
```

**LLM** (gpt-oss:120b, 2.2 s, passed the checks)

```
Findings:
- A left-hemisphere frontal region tumor is identified (estimate).
- Total tumor volume 24.8 cm³ (doctor-corrected mask): enhancing component 9.1 cm³, necrotic/non‑enhancing core 1.2 cm³, surrounding edema 14.5 cm³.
- 2D classifier indicates glioma (62%, not calibrated).
- Reviewing physician designates tumor type as meningioma.

Impression: Meningioma measuring approximately 25 cm³ in the left frontal region (estimate).

Notes:
- The region is a rough position estimate, not an atlas label.

AI-assisted draft from a research prototype, not a medical device. The reviewing doctor makes the diagnosis.
```
