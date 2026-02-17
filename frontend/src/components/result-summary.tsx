"use client";

import { useEffect, useState } from "react";

import { apiFileUrl, TUMOR_LABELS, type ModelCard, type ResultOut } from "@/lib/api";

// Same colors as the mask overlay in the viewer (BraTS convention), so a region reads the same everywhere
const PARTS = [
  { key: "edema", label: "Edema", swatch: "bg-green-500" },
  { key: "necrosis_non_enhancing", label: "Necrotic / non-enhancing core", swatch: "bg-red-500" },
  { key: "enhancing", label: "Enhancing tumor", swatch: "bg-yellow-400" },
] as const;

const cm3 = (v: number | undefined) => (v === undefined ? "—" : `${v.toFixed(1)} cm³`);
const pct = (p: number) => `${Math.round(p * 100)}%`;
const typeLabel = (t: string) => TUMOR_LABELS[t] ?? t;

export function ResultSummary({ caseId, result, reviewed, token, gradcam = true }: {
  caseId: string;
  result: ResultOut;
  reviewed: boolean;
  token: string;
  gradcam?: boolean;  // false when the page shows the heatmap elsewhere (2D cases: in the main area)
}) {
  // After the doctor fixed the mask, its volumes are the ones that count; the AI's stay for comparison
  const corrected = result.info?.corrected_mask?.volumes_cm3 ?? null;
  const volumes = corrected ?? result.volumes_cm3;
  const is3d = volumes !== null;
  const found = is3d && (volumes.whole_tumor ?? 0) > 0;
  const probs = Object.entries(result.probabilities ?? {}).sort((a, b) => b[1] - a[1]);
  // The 2D model sees one slice and was trained on another dataset; when it contradicts the 3D outline,
  // show the conflict instead of a confident type (same rule as the report notes)
  const typed = result.tumor_type !== null && result.tumor_type !== "notumor";
  const disagree = is3d && result.tumor_type !== null && typed !== found;
  const typeNote = result.confidence === null ? undefined
    : disagree ? `2D model: ${typeLabel(result.tumor_type!)} ${pct(result.confidence)} · ${found ? "3D outlined a tumor" : "no 3D outline"}`
    : `${pct(result.confidence)} from the 2D model`;
  const where = result.side
    ? `${result.side[0].toUpperCase()}${result.side.slice(1)}${result.lobe_estimate ? ` · ${result.lobe_estimate} (est.)` : ""}`
    : is3d ? "No tumor found" : "Not measured";

  return (
    <section className="flex flex-col gap-5" aria-labelledby="result-title">
      <h2 id="result-title" className="font-semibold">
        AI result <span className="font-normal text-muted-foreground">· {reviewed ? "reviewed by the doctor" : "a draft until you approve it"}</span>
      </h2>

      <dl className="grid grid-cols-[repeat(auto-fit,minmax(150px,1fr))] gap-3">
        <Fact label="Tumor type" value={disagree ? "Models disagree" : result.tumor_type ? typeLabel(result.tumor_type) : "—"}
              note={typeNote} />
        <Fact label="Whole tumor" value={!is3d ? "Not measured" : found ? cm3(volumes.whole_tumor) : "None found"}
              note={!is3d ? "A 2D slice has no volume" : corrected
                ? `Your corrected mask · AI: ${cm3(result.volumes_cm3?.whole_tumor)}` : "3D segmentation"} />
        <Fact label="Location" value={where} note={result.side ? "Rough position rule; confirm on the scan" : undefined} />
      </dl>

      <div className="grid grid-cols-[repeat(auto-fit,minmax(280px,1fr))] gap-6">
        {is3d && !found && (
          <div className="flex flex-col gap-3">
            <h3 className="font-semibold">Volumes</h3>
            <p className="text-muted-foreground">The 3D model outlined no tumor in this scan. Check the images yourself.</p>
          </div>
        )}
        {found && (
          <div className="flex flex-col gap-3">
            <h3 className="font-semibold">
              Volumes{corrected && <span className="font-normal text-muted-foreground"> · from your corrected mask</span>}
            </h3>
            <table className="text-[15px]">
              <caption className="sr-only">Tumor volume per region in cubic centimeters</caption>
              <tbody>
                {PARTS.map((p) => (
                  <tr key={p.key} className="border-b">
                    <th scope="row" className="py-2 text-left font-normal">
                      <span className={`mr-2.5 inline-block size-2.5 rounded-full ${p.swatch}`} aria-hidden />
                      {p.label}
                    </th>
                    <td className="py-2 text-right tabular-nums">{cm3(volumes[p.key])}</td>
                  </tr>
                ))}
                <tr className="border-b">
                  <th scope="row" className="py-2 text-left font-normal text-muted-foreground">Tumor core (red + yellow)</th>
                  <td className="py-2 text-right tabular-nums text-muted-foreground">{cm3(volumes.tumor_core)}</td>
                </tr>
                <tr>
                  <th scope="row" className="py-2 text-left font-semibold">Whole tumor</th>
                  <td className="py-2 text-right font-semibold tabular-nums">{cm3(volumes.whole_tumor)}</td>
                </tr>
              </tbody>
            </table>
          </div>
        )}

        {probs.length > 0 && (
          <div className="flex flex-col gap-3">
            <h3 className="font-semibold">Tumor type, all classes</h3>
            <ul className="flex flex-col gap-2.5" aria-label="Model probability per tumor type">
              {probs.map(([t, p], i) => (
                <li key={t} className="grid grid-cols-[7.5rem_1fr_3rem] items-center gap-3 text-[15px]">
                  <span className={i === 0 ? "font-semibold" : ""}>{typeLabel(t)}</span>
                  <span className="h-2 rounded-full bg-muted" aria-hidden>
                    {/* Top class in the accent, the rest recede; values are printed, so color is never the only cue */}
                    <span className={`block h-full rounded-full ${i === 0 ? "bg-blue-600" : "bg-zinc-400"}`}
                          style={{ width: `${Math.max(p * 100, 1)}%` }} />
                  </span>
                  <span className="text-right tabular-nums">{pct(p)}</span>
                </li>
              ))}
            </ul>
            <p className="text-sm text-muted-foreground">
              From the 2D model. Input: {result.info?.classifier_input ?? "one slice"}. Percentages are not calibrated.
            </p>
          </div>
        )}
      </div>

      {gradcam && result.has_gradcam && <GradCam caseId={caseId} result={result} token={token} />}

      <div className="flex flex-col gap-1 border-t pt-4 text-sm text-muted-foreground">
        <p className="font-medium text-foreground">
          Research prototype, not a medical device. Check the scan yourself; the decision is yours.
        </p>
        <ModelList result={result} />
      </div>
    </section>
  );
}

function Fact({ label, value, note }: { label: string; value: string; note?: string }) {
  return (
    <div className="flex flex-col gap-1 rounded-lg bg-muted/60 px-3.5 py-3">
      <dt className="text-sm text-muted-foreground">{label}</dt>
      <dd className="text-lg font-semibold">{value}</dd>
      {note && <dd className="text-sm text-muted-foreground">{note}</dd>}
    </div>
  );
}

export function GradCam({ caseId, result, token, large = false }: {
  caseId: string;
  result: ResultOut;
  token: string;
  large?: boolean;  // the main image of a 2D case
}) {
  const [src, setSrc] = useState<string | null>(null);
  const [failed, setFailed] = useState(false);

  useEffect(() => {
    let url: string | null = null;
    let live = true;
    apiFileUrl(`/cases/${caseId}/results/${result.id}/gradcam`, token).then(
      (u) => {
        url = u;
        if (live) setSrc(u);
        else URL.revokeObjectURL(u);
      },
      () => live && setFailed(true),
    );
    return () => {
      live = false;
      if (url) URL.revokeObjectURL(url);
    };
  }, [caseId, result.id, token]);

  return (
    <figure className={large ? "flex h-full min-h-0 flex-col items-center gap-2 p-4 text-zinc-100" : "flex flex-col gap-2"}>
      <h3 className="font-semibold">Where the 2D model looked (Grad-CAM)</h3>
      <div className={`flex aspect-square items-center justify-center overflow-hidden rounded-xl bg-black ${
        large ? "min-h-0 w-auto max-w-full flex-1" : "w-full max-w-sm"}`}>
        {src ? (
          // eslint-disable-next-line @next/next/no-img-element -- a local blob URL; next/image can't optimize it
          <img src={src} alt="Grad-CAM heatmap over the MRI slice the 2D model classified" className="size-full object-contain" />
        ) : (
          <span className="text-sm text-zinc-400">{failed ? "Heatmap could not be loaded." : "Loading heatmap…"}</span>
        )}
      </div>
      <figcaption className={`max-w-sm text-sm ${large ? "text-zinc-400" : "text-muted-foreground"}`}>
        Red areas mattered most for the tumor type. It is a heatmap, not the tumor outline.
      </figcaption>
    </figure>
  );
}

const MODEL_ROLES: Record<string, string> = { segmenter: "3D segmenter", classifier: "2D classifier" };

/** Which model versions made this result, as read from the model files that ran. */
function ModelList({ result }: { result: ResultOut }) {
  const models = Object.entries(result.info?.models ?? {}) as [string, ModelCard][];
  if (!models.length) return <p>Models: {result.model_version || "—"}</p>;
  return (
    <ul className="flex flex-col gap-0.5">
      {models.map(([role, m]) => (
        <li key={role}>
          {MODEL_ROLES[role] ?? role}: <span className="font-mono text-foreground">{m.version}</span>
          {m.about && <span> · {m.about}</span>}
        </li>
      ))}
    </ul>
  );
}
