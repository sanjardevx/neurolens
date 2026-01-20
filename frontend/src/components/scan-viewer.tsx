"use client";

import type { Niivue } from "@niivue/niivue";
import { useEffect, useRef, useState } from "react";

import { API_URL, upload } from "@/lib/api";

const MRI = [
  ["flair", "FLAIR"],
  ["t1", "T1"],
  ["t1ce", "T1ce"],
  ["t2", "T2"],
] as const;
type Mri = (typeof MRI)[number][0];

// NiiVue's SLICE_TYPE values; the axis is the RAS index that a single view scrolls through
const VIEWS = [
  { id: "axial", label: "Axial", type: 0, axis: 2 },
  { id: "coronal", label: "Coronal", type: 1, axis: 1 },
  { id: "sagittal", label: "Sagittal", type: 2, axis: 0 },
  { id: "all", label: "All 3", type: 3, axis: null },
] as const;
type View = (typeof VIEWS)[number];

// BraTS mask labels; colors match the result summary (Tailwind red-500, green-500, yellow-400)
const REGIONS = [
  { id: "edema", label: "Edema", value: 2, rgb: [34, 197, 94], swatch: "bg-green-500" },
  { id: "core", label: "Necrotic core", value: 1, rgb: [239, 68, 68], swatch: "bg-red-500" },
  { id: "enhancing", label: "Enhancing", value: 4, rgb: [250, 204, 21], swatch: "bg-yellow-400" },
] as const;
type Region = (typeof REGIONS)[number]["id"];

/** Label lookup table for NiiVue: one entry per label value 0-4; hidden regions get alpha 0. */
function maskColors(shown: Record<Region, boolean>) {
  const lut = { R: [0, 0, 0, 0, 0], G: [0, 0, 0, 0, 0], B: [0, 0, 0, 0, 0], A: [0, 0, 0, 0, 0], I: [0, 1, 2, 3, 4] };
  for (const r of REGIONS) {
    [lut.R[r.value], lut.G[r.value], lut.B[r.value]] = r.rgb;
    lut.A[r.value] = shown[r.id] ? 255 : 0;
  }
  return lut;
}

const PEN_SIZES = [
  ["S", 1],
  ["M", 3],
  ["L", 7],
] as const;
type Tool = Region | "erase";

/** Dark reading-room viewer for one stored 3D scan: pick the MRI type and the plane, scroll the slices,
 *  lay the AI mask on top (per region, with opacity) and, for the case's doctor, fix the mask with a brush. */
export function ScanViewer({ caseId, scanId, resultId, maskKind, canEdit, onMaskSaved, token }: {
  caseId: string;
  scanId: string;
  resultId: string | null;  // the result whose mask to overlay; null while there is none
  maskKind: "mask" | "corrected";  // the AI's mask, or the doctor's saved correction
  canEdit: boolean;
  onMaskSaved: () => void;
  token: string;
}) {
  const canvas = useRef<HTMLCanvasElement>(null);
  const nvRef = useRef<Niivue | null>(null);
  const [nvReady, setNvReady] = useState(false);
  const [mri, setMri] = useState<Mri>("flair");
  const [view, setView] = useState<View>(VIEWS[0]);
  const [state, setState] = useState<"loading" | "ready" | "error">("loading");
  const [slice, setSlice] = useState<{ i: number; n: number } | null>(null);
  const [maskOn, setMaskOn] = useState(true);
  const [shown, setShown] = useState<Record<Region, boolean>>({ edema: true, core: true, enhancing: true });
  const [opacity, setOpacity] = useState(0.6);
  const [editing, setEditing] = useState(false);
  const [tool, setTool] = useState<Tool>("edema");
  const [penSize, setPenSize] = useState<number>(3);
  const [saving, setSaving] = useState(false);
  const [editError, setEditError] = useState<string | null>(null);

  // WebGL needs the browser, so NiiVue is loaded and attached only after the page is on screen
  useEffect(() => {
    let nv: Niivue | null = null;
    let gone = false;
    import("@niivue/niivue").then(async ({ Niivue }) => {
      if (gone || !canvas.current) return;
      nv = new Niivue({
        backColor: [0, 0, 0, 1],
        crosshairColor: [0.23, 0.51, 0.96, 1],
        crosshairWidth: 1,
        isRadiologicalConvention: true,  // patient's right on the screen's left, as radiologists read
        isOrientationTextVisible: true,
        dragAndDropEnabled: false,
        show3Dcrosshair: false,
        multiplanarPadPixels: 4,
        multiplanarShowRender: 0,  // SHOW_RENDER.NEVER: three slice planes only, no 3D render tile
        logLevel: "error",
        loadingText: "",
      });
      await nv.attachToCanvas(canvas.current);
      nvRef.current = nv;
      setNvReady(true);
    });
    return () => {
      gone = true;
      nv?.cleanup();
      nvRef.current = null;
    };
  }, []);

  // Load the chosen MRI type; keep the crosshair where the doctor was looking
  useEffect(() => {
    const nv = nvRef.current;
    if (!nvReady || !nv) return;
    let live = true;
    const keep = nv.volumes.length ? [...nv.scene.crosshairPos] : null;
    setState("loading");
    const headers = { Authorization: `Bearer ${token}` };
    nv.loadVolumes([
      // name with the extension tells NiiVue the file format; the URL itself has none
      { url: `${API_URL}/cases/${caseId}/scans/${scanId}/${mri}`, name: `${mri}.nii.gz`, headers },
      ...(resultId ? [{ url: `${API_URL}/cases/${caseId}/results/${resultId}/${maskKind}`, name: "mask.nii.gz", headers }] : []),
    ]).then(
      () => {
        if (!live) return;
        if (keep) nv.scene.crosshairPos = keep as typeof nv.scene.crosshairPos;
        nv.drawScene();
        setState("ready");
      },
      () => live && setState("error"),
    );
    return () => {
      live = false;
    };
  }, [nvReady, mri, caseId, scanId, resultId, maskKind, token]);

  // Mask look: which regions, how opaque, on or off. The mask is volume 1 when it exists.
  useEffect(() => {
    const nv = nvRef.current;
    const mask = nv?.volumes[1];
    if (!nv || !mask || state !== "ready") return;
    mask.setColormapLabel(maskColors(shown));
    // While editing, the drawing layer shows the mask instead; the overlay volume would double it
    nv.setOpacity(1, maskOn && !editing ? opacity : 0);
    nv.setDrawOpacity(opacity);
    nv.updateGLVolume();
  }, [state, maskOn, shown, opacity, editing]);

  // Brush: pen value = BraTS label (0 erases); size in voxels on the slice being drawn
  useEffect(() => {
    const nv = nvRef.current;
    if (!nv || !editing) return;
    nv.opts.penSize = penSize;
    nv.setPenValue(tool === "erase" ? 0 : REGIONS.find((r) => r.id === tool)!.value, false);
  }, [editing, tool, penSize]);

  function startEditing() {
    const nv = nvRef.current;
    if (!nv?.volumes[1]) return;
    // The current mask becomes NiiVue's drawing layer, painted with the same colors as the overlay
    nv.loadDrawing(nv.volumes[1]);
    nv.setDrawColormap(maskColors({ edema: true, core: true, enhancing: true }) as unknown as string);
    nv.setDrawingEnabled(true);
    setMaskOn(true);
    setEditError(null);
    setEditing(true);
  }

  function stopEditing() {
    const nv = nvRef.current;
    nv?.setDrawingEnabled(false);
    nv?.closeDrawing();
    setEditing(false);
  }

  async function saveMask() {
    const nv = nvRef.current;
    if (!nv || !resultId) return;
    setSaving(true);
    setEditError(null);
    try {
      // NIfTI of the drawing in the scan's own orientation; the API turns it back onto the mask grid
      const nifti = await nv.saveImage({ filename: "", isSaveDrawing: true, volumeByIndex: 0 });
      if (!(nifti instanceof Uint8Array)) throw new Error("Could not read the drawing.");
      const form = new FormData();
      form.append("mask", new Blob([nifti as BlobPart]), "corrected.nii");
      await upload(`/cases/${caseId}/results/${resultId}/mask`, token, form, () => {});
      stopEditing();
      onMaskSaved();
    } catch (e) {
      setEditError(e instanceof Error ? `The mask was not saved: ${e.message}` : "The mask was not saved.");
    } finally {
      setSaving(false);
    }
  }

  // Plane change, and keep the slice counter in sync with mouse-wheel scrolling inside the canvas
  useEffect(() => {
    const nv = nvRef.current;
    if (!nv || state !== "ready") return;
    nv.setSliceType(view.type);
    const sync = () => {
      const dims = nv.volumes[0]?.dimsRAS;
      if (view.axis === null || !dims) return setSlice(null);
      const vox = nv.frac2vox(nv.scene.crosshairPos);
      setSlice({ i: Math.round(vox[view.axis]) + 1, n: dims[view.axis + 1] });
    };
    sync();
    nv.onLocationChange = sync;
  }, [view, state]);

  function goToSlice(i: number) {
    const nv = nvRef.current;
    if (!nv || view.axis === null || !slice) return;
    nv.scene.crosshairPos[view.axis] = (i - 0.5) / slice.n;
    nv.drawScene();
    setSlice({ ...slice, i });
  }

  return (
    <section aria-label="Scan viewer" className="flex flex-col gap-3 rounded-xl bg-zinc-950 p-3 text-zinc-100 sm:p-4">
      <div className="flex flex-wrap items-center justify-between gap-2">
        <Segmented label="MRI type" options={MRI.map(([id, l]) => [id, l])} value={mri} disabled={editing}
                   onPick={(v) => setMri(v as Mri)} />
        <Segmented label="Plane" options={VIEWS.map((v) => [v.id, v.label])} value={view.id}
                   onPick={(id) => setView(VIEWS.find((v) => v.id === id)!)} />
      </div>

      <div className="relative aspect-square w-full overflow-hidden rounded-lg bg-black sm:aspect-[4/3]">
        <canvas ref={canvas} className="size-full" aria-label={`${mri.toUpperCase()} MRI, ${view.label} view`} />
        {state !== "ready" && (
          <p role="status" className="absolute inset-0 flex items-center justify-center text-sm text-zinc-400">
            {state === "loading" ? "Loading the scan…" : "The scan could not be loaded."}
          </p>
        )}
      </div>

      {slice ? (
        <label className="flex items-center gap-3 text-sm text-zinc-300">
          <span className="w-28 shrink-0 tabular-nums">Slice {slice.i} / {slice.n}</span>
          <input
            type="range"
            min={1}
            max={slice.n}
            value={slice.i}
            onChange={(e) => goToSlice(Number(e.target.value))}
            className="w-full accent-blue-500"
            aria-label={`${view.label} slice`}
          />
        </label>
      ) : (
        <p className="text-sm text-zinc-400">Click a view to move the crosshair; scroll inside a view to change its slice.</p>
      )}
      {resultId && (
        <div className="flex flex-wrap items-center gap-x-5 gap-y-3 border-t border-zinc-800 pt-3 text-sm">
          <button
            type="button"
            aria-pressed={maskOn}
            disabled={editing}
            onClick={() => setMaskOn(!maskOn)}
            className={`min-h-9 rounded-md px-3 font-semibold ${maskOn ? "bg-zinc-100 text-zinc-950" : "bg-zinc-900 text-zinc-300"}`}
          >
            AI mask {maskOn ? "on" : "off"}
          </button>
          <fieldset className="flex flex-wrap gap-x-4 gap-y-2" disabled={!maskOn || editing}>
            <legend className="sr-only">Mask regions</legend>
            {REGIONS.map((r) => (
              <label key={r.id} className={`flex min-h-9 items-center gap-2 ${maskOn && !editing ? "text-zinc-200" : "text-zinc-600"}`}>
                <input
                  type="checkbox"
                  checked={shown[r.id]}
                  onChange={(e) => setShown({ ...shown, [r.id]: e.target.checked })}
                  className="size-4 accent-blue-500"
                />
                <span className={`size-2.5 rounded-full ${r.swatch}`} aria-hidden />
                {r.label}
              </label>
            ))}
          </fieldset>
          <label className={`flex min-w-48 flex-1 items-center gap-3 ${maskOn ? "text-zinc-300" : "text-zinc-600"}`}>
            <span className="shrink-0">Opacity</span>
            <input
              type="range"
              min={0.1}
              max={1}
              step={0.05}
              value={opacity}
              disabled={!maskOn}
              onChange={(e) => setOpacity(Number(e.target.value))}
              className="w-full accent-blue-500"
              aria-label="Mask opacity"
            />
            <span className="w-10 text-right tabular-nums">{Math.round(opacity * 100)}%</span>
          </label>
        </div>
      )}
      {resultId && !editing && (
        <div className="flex flex-wrap items-center gap-3 text-sm text-zinc-400">
          <span>{maskKind === "corrected" ? "Showing your corrected mask." : "Showing the AI mask."}</span>
          {canEdit && (
            <button type="button" onClick={startEditing} disabled={state !== "ready"}
                    className="min-h-9 rounded-md border border-zinc-700 px-3 font-semibold text-zinc-100 hover:bg-zinc-900">
              Fix the mask
            </button>
          )}
        </div>
      )}
      {editing && (
        <div className="flex flex-col gap-3 rounded-lg border border-blue-500/40 bg-zinc-900 p-3 text-sm">
          <div className="flex flex-wrap items-center gap-2">
            <Segmented label="Brush" value={tool} onPick={(t) => setTool(t as Tool)}
                       options={[...REGIONS.map((r) => [r.id, r.label] as const), ["erase", "Eraser"] as const]} />
            <Segmented label="Brush size" value={String(penSize)} onPick={(v) => setPenSize(Number(v))}
                       options={PEN_SIZES.map(([l, n]) => [String(n), l] as const)} />
            <button type="button" onClick={() => nvRef.current?.drawUndo()}
                    className="min-h-9 rounded-md px-3 font-semibold text-zinc-300 hover:bg-zinc-800">
              Undo
            </button>
          </div>
          <p className="text-zinc-400">Drag on a slice to paint; scroll to move between slices. The AI&apos;s mask is kept as it was.</p>
          {editError && <p role="alert" className="rounded-md bg-red-500/15 px-3 py-2 text-red-300">{editError}</p>}
          <div className="flex flex-wrap gap-2">
            <button type="button" onClick={saveMask} disabled={saving}
                    className="min-h-10 rounded-md bg-blue-600 px-4 font-semibold text-white hover:bg-blue-500 disabled:opacity-60">
              {saving ? "Saving…" : "Save mask"}
            </button>
            <button type="button" onClick={stopEditing} disabled={saving}
                    className="min-h-10 rounded-md px-4 font-semibold text-zinc-300 hover:bg-zinc-800">
              Cancel
            </button>
          </div>
        </div>
      )}
      <p className="text-xs text-zinc-500">Radiological view: the patient&apos;s right is on the left (R / L marks on the image).</p>
    </section>
  );
}

function Segmented({ label, options, value, onPick, disabled }: {
  label: string;
  options: (readonly [string, string])[];
  value: string;
  onPick: (id: string) => void;
  disabled?: boolean;
}) {
  return (
    <div role="group" aria-label={label} className="flex rounded-lg bg-zinc-900 p-1">
      {options.map(([id, text]) => (
        <button
          key={id}
          type="button"
          aria-pressed={value === id}
          disabled={disabled}
          onClick={() => onPick(id)}
          className={`min-h-9 rounded-md px-3 text-[13px] font-semibold ${
            value === id ? "bg-zinc-100 text-zinc-950" : "text-zinc-400 hover:text-zinc-100"
          } disabled:opacity-50`}
        >
          {text}
        </button>
      ))}
    </div>
  );
}
