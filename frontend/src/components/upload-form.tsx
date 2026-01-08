"use client";

import { useState, type DragEvent } from "react";

import { Button } from "@/components/ui/button";
import { ApiError, upload } from "@/lib/api";

const MRI = [
  ["flair", "FLAIR"],
  ["t1", "T1"],
  ["t1ce", "T1ce"],
  ["t2", "T2"],
] as const;
type Mri = (typeof MRI)[number][0];
type Mode = "3d" | "2d";

/** Guess the MRI type from a file name like BraTS2021_00002_t1ce.nii.gz or t1_series.zip. */
function mriTypeOf(name: string): Mri | null {
  const words = name.toLowerCase().split(/[^a-z0-9]+/);
  if (words.some((w) => ["t1ce", "t1c", "t1gd", "t1post"].includes(w))) return "t1ce";
  if (words.includes("flair")) return "flair";
  if (words.includes("t2")) return "t2";
  if (words.includes("t1")) return "t1";
  return null;
}

const is3dFile = (f: File) => /\.(nii|nii\.gz|zip)$/i.test(f.name);
const is2dFile = (f: File) => /\.(png|jpe?g)$/i.test(f.name);
const mb = (f: File) => `${(f.size / 1e6).toFixed(1)} MB`;

export function UploadForm({ caseId, token, onUploaded }: { caseId: string; token: string; onUploaded: () => void }) {
  const [mode, setMode] = useState<Mode>("3d");
  const [mri, setMri] = useState<Partial<Record<Mri, File>>>({});
  const [image, setImage] = useState<File | null>(null);
  const [share, setShare] = useState<number | null>(null);  // upload progress 0-1; null when idle
  const [error, setError] = useState<string | null>(null);

  const busy = share !== null;
  const ready = mode === "3d" ? MRI.every(([m]) => mri[m]) : image !== null;

  function addFiles(files: File[], only?: Mri) {
    setError(null);
    if (mode === "2d") {
      const f = files[0];
      if (f && !is2dFile(f)) return setError(`${f.name}: choose a JPG or PNG slice.`);
      return setImage(f ?? null);
    }
    const next = { ...mri };
    const unknown: string[] = [];
    for (const f of files) {
      if (!is3dFile(f)) return setError(`${f.name}: choose .nii, .nii.gz or a .zip of DICOM slices.`);
      const m = only ?? mriTypeOf(f.name);
      if (m) next[m] = f;
      else unknown.push(f.name);
    }
    setMri(next);
    if (unknown.length) setError(`Couldn't tell the MRI type of ${unknown.join(", ")}. Choose it in its row.`);
  }

  function onDrop(e: DragEvent) {
    e.preventDefault();
    if (!busy) addFiles(Array.from(e.dataTransfer.files));
  }

  async function start() {
    const form = new FormData();
    if (mode === "3d") MRI.forEach(([m]) => form.append(m, mri[m]!));
    else form.append("image", image!);
    setError(null);
    setShare(0);
    try {
      await upload(`/cases/${caseId}/scans`, token, form, setShare);
      onUploaded();
    } catch (e) {
      // 422 = the scan failed a quality check; the API says why in plain words
      const why = e instanceof Error ? e.message : "Upload failed";
      setError(e instanceof ApiError && e.status === 422 ? `The scan was not accepted: ${why}.` : why);
      setShare(null);
    }
  }

  return (
    <section
      onDragOver={(e) => e.preventDefault()}
      onDrop={onDrop}
      className="flex flex-col gap-5 rounded-xl border bg-background p-5 sm:p-6"
      aria-labelledby="upload-title"
    >
      <div className="flex flex-wrap items-center justify-between gap-3">
        <h2 id="upload-title" className="text-lg font-semibold">Upload the scan</h2>
        <div role="group" aria-label="Scan type" className="flex rounded-xl bg-muted p-1">
          {(["3d", "2d"] as const).map((m) => (
            <button
              key={m}
              type="button"
              disabled={busy}
              aria-pressed={mode === m}
              onClick={() => {
                setMode(m);
                setError(null);
              }}
              className={`min-h-10 rounded-lg px-4 text-sm font-semibold ${
                mode === m ? "bg-background shadow-sm" : "text-muted-foreground hover:text-foreground"
              }`}
            >
              {m === "3d" ? "3D MRI" : "2D slice"}
            </button>
          ))}
        </div>
      </div>

      {mode === "3d" ? (
        <>
          <ul className="flex flex-col divide-y rounded-lg border">
            {MRI.map(([m, label]) => {
              const f = mri[m];
              return (
                <li key={m} className="flex flex-wrap items-center gap-x-4 gap-y-2 px-4 py-3">
                  <span className="w-14 font-semibold">{label}</span>
                  <span className={`min-w-0 flex-1 truncate ${f ? "" : "text-muted-foreground"}`}>
                    {f ? `${f.name} · ${mb(f)}` : "Not chosen"}
                  </span>
                  <FilePick
                    label={f ? "Change" : "Choose file"}
                    name={`${label} file`}
                    accept=".nii,.gz,.zip"
                    disabled={busy}
                    onPick={(files) => addFiles(files, m)}
                  />
                </li>
              );
            })}
          </ul>
          <div className="flex flex-wrap items-center gap-3">
            <FilePick label="Choose all 4 at once" name="all 4 MRI files" accept=".nii,.gz,.zip" multiple disabled={busy} onPick={addFiles} />
            <p className="text-sm text-muted-foreground">or drop them here. Names like …_t1ce.nii.gz are matched by type.</p>
          </div>
          <p className="text-sm text-muted-foreground">
            Each type: .nii, .nii.gz or a .zip of one DICOM series. Patient names, IDs and dates are removed before saving.
          </p>
        </>
      ) : (
        <>
          <div className="flex flex-wrap items-center gap-x-4 gap-y-2 rounded-lg border px-4 py-3">
            <span className={`min-w-0 flex-1 truncate ${image ? "" : "text-muted-foreground"}`}>
              {image ? `${image.name} · ${mb(image)}` : "No slice chosen"}
            </span>
            <FilePick label={image ? "Change" : "Choose file"} name="MRI slice" accept="image/png,image/jpeg" disabled={busy} onPick={addFiles} />
          </div>
          <p className="text-sm text-muted-foreground">
            One axial MRI slice, JPG or PNG. You get the tumor type and a heatmap, no 3D volumes.
          </p>
        </>
      )}

      {error && (
        <p role="alert" className="rounded-lg bg-destructive/10 px-3 py-2 text-sm text-destructive">{error}</p>
      )}

      {busy ? (
        <div className="flex flex-col gap-2" role="status">
          <Progress value={share} label="Upload progress" />
          <p className="text-sm text-muted-foreground">
            {share < 1 ? `Uploading… ${Math.round(share * 100)}%` : "Checking the scan and removing patient details…"}
          </p>
        </div>
      ) : (
        <div className="flex flex-wrap items-center gap-4">
          <Button onClick={start} disabled={!ready} className="h-12 px-6 text-base font-semibold">
            Start AI analysis
          </Button>
          <span className="text-sm text-muted-foreground">
            {mode === "3d" ? "About 15 seconds after upload." : "About 1 second."}
          </span>
        </div>
      )}
    </section>
  );
}

function FilePick(props: {
  label: string;
  name: string;
  accept: string;
  multiple?: boolean;
  disabled?: boolean;
  onPick: (files: File[]) => void;
}) {
  // A real <input type=file> inside the label keeps it keyboard and screen-reader friendly
  return (
    <label
      className={`inline-flex min-h-10 cursor-pointer items-center rounded-lg border px-4 text-sm font-semibold hover:bg-muted has-focus-visible:ring-3 has-focus-visible:ring-ring/50 ${
        props.disabled ? "pointer-events-none opacity-50" : ""
      }`}
    >
      {props.label}
      <input
        type="file"
        className="sr-only"
        aria-label={`Choose ${props.name}`}
        accept={props.accept}
        multiple={props.multiple}
        disabled={props.disabled}
        onChange={(e) => {
          props.onPick(Array.from(e.target.files ?? []));
          e.target.value = "";  // picking the same file again still fires onChange
        }}
      />
    </label>
  );
}

export function Progress({ value, label }: { value: number; label: string }) {
  return (
    <div
      role="progressbar"
      aria-label={label}
      aria-valuemin={0}
      aria-valuemax={100}
      aria-valuenow={Math.round(value * 100)}
      className="h-2.5 overflow-hidden rounded-full bg-muted"
    >
      <div className="h-full rounded-full bg-blue-600 transition-[width] duration-300" style={{ width: `${value * 100}%` }} />
    </div>
  );
}
