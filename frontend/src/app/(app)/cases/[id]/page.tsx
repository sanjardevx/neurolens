"use client";

import { ArrowLeft } from "lucide-react";
import Link from "next/link";
import { useParams } from "next/navigation";
import { type ReactNode, useEffect, useState } from "react";

import { useSession } from "@/components/session";
import { GradCam, ResultSummary } from "@/components/result-summary";
import { ReviewPanel } from "@/components/review-panel";
import { ScanViewer } from "@/components/scan-viewer";
import { Progress, UploadForm } from "@/components/upload-form";
import { ErrorNote, Skeleton } from "@/components/states";
import { api, type ApiError, type CaseOut, type ResultOut, type ReviewOut, type ScanOut } from "@/lib/api";
import { statusOf, whenOf } from "@/lib/cases";

const POLL_MS = 2000;

const TABS = [
  ["findings", "Findings"],
  ["decision", "Report and decision"],
] as const;
type Tab = (typeof TABS)[number][0];

export default function CasePage() {
  const { id } = useParams<{ id: string }>();
  const { token, role } = useSession();
  const [kase, setCase] = useState<CaseOut | null>(null);
  const [result, setResult] = useState<ResultOut | null>(null);
  const [scans, setScans] = useState<ScanOut[]>([]);
  const [review, setReview] = useState<ReviewOut | null>(null);
  const [error, setError] = useState<ApiError | null>(null);
  const [offline, setOffline] = useState(false);  // a progress poll failed; it keeps trying
  const [version, setVersion] = useState(0);  // bump to read the case again (after upload, when the AI ends)
  const [tab, setTab] = useState<Tab>("findings");

  useEffect(() => {
    let live = true;
    Promise.all([
      api<CaseOut>(`/cases/${id}`, token),
      api<ResultOut[]>(`/cases/${id}/results`, token),
      api<ScanOut[]>(`/cases/${id}/scans`, token),
      api<ReviewOut[]>(`/cases/${id}/reviews`, token),
    ]).then(
      ([c, results, s, reviews]) => {
        if (!live) return;
        setCase(c);
        setResult(results[0] ?? null);
        setScans(s);
        setReview(reviews[0] ?? null);  // newest first
        setError(null);
        setOffline(false);
      },
      (e: ApiError) => live && setError(e),
    );
    return () => {
      live = false;
    };
  }, [id, token, version]);

  // While the AI works, poll only the results: each case read is written to the audit log
  const working = kase?.status === "processing";
  useEffect(() => {
    if (!working) return;
    let live = true;
    const timer = setInterval(() => {
      api<ResultOut[]>(`/cases/${id}/results`, token).then(
        ([r]) => {
          if (!live || !r) return;
          setOffline(false);
          setResult(r);
          if (r.progress === 100 || r.error) setVersion((v) => v + 1);
        },
        () => live && setOffline(true),
      );
    }, POLL_MS);
    return () => {
      live = false;
      clearInterval(timer);
    };
  }, [working, id, token]);

  if (!kase) {
    return (
      <div className="flex flex-col gap-6 px-4 py-9 sm:px-10">
        <Link href="/" className="w-fit text-sm font-medium text-blue-700 hover:text-blue-800">← Worklist</Link>
        {error?.status === 404 ? (
          <div className="flex flex-col gap-2">
            <h1 className="text-2xl font-bold">Case not found</h1>
            {/* The API answers 404 for another doctor's case too, so nobody learns which codes exist */}
            <p className="text-muted-foreground">It doesn&apos;t exist, or it belongs to another doctor.</p>
          </div>
        ) : error ? (
          <ErrorNote message={error.message} onRetry={() => setVersion((v) => v + 1)} />
        ) : (
          <Skeleton label="Loading case…" lines={4} />
        )}
      </div>
    );
  }

  const [label, pill] = statusOf({ status: kase.status, progress: result?.progress ?? null, decision: review?.decision ?? null });
  const canUpload = role === "doctor" && (kase.status === "new" || kase.status === "failed");
  const done = result !== null && (kase.status === "ready" || kase.status === "reviewed");
  // The scan the latest result was made from (or the newest upload); only 3D scans open in the viewer
  const scan = scans.find((s) => s.id === result?.scan_id) ?? scans.toSorted((a, b) => b.created_at.localeCompare(a.created_at))[0];
  const showViewer = scan && scan.file_format !== "image" && kase.status !== "new";
  const gradcamMain = !showViewer && done && result.has_gradcam;  // a 2D case: the heatmap is the image
  const fixed = result?.info?.corrected_mask?.volumes_cm3?.whole_tumor;

  const image = showViewer ? (
    <ScanViewer
      caseId={kase.id}
      scanId={scan.id}
      token={token}
      resultId={result?.has_mask && result.scan_id === scan.id ? result.id : null}
      maskKind={result?.has_corrected_mask ? "corrected" : "mask"}
      canEdit={role === "doctor" && kase.status === "ready"}
      onMaskSaved={() => setVersion((v) => v + 1)}
      volumes={{ ai: result?.volumes_cm3?.whole_tumor, fixed }}
    />
  ) : gradcamMain ? (
    <GradCam caseId={kase.id} result={result} token={token} large />
  ) : null;

  const alerts = (
    <>
      {(offline || error) && (
        <ErrorNote
          message={`${error?.message ?? "Lost the connection to the API."} Showing the last loaded state.`}
          onRetry={() => setVersion((v) => v + 1)}
        />
      )}
      {kase.status === "failed" && result?.error && (
        <p role="alert" className="rounded-lg bg-destructive/10 px-4 py-3 text-destructive">
          The AI could not analyze this scan ({result.error}). Check the files and upload them again.
        </p>
      )}
    </>
  );

  const progress = working && (
    <section className="flex flex-col gap-3" role="status">
      <h2 className="font-semibold">AI is analyzing the scan</h2>
      <Progress value={(result?.progress ?? 0) / 100} label="AI progress" />
      <p className="text-sm text-muted-foreground">
        {result?.progress ?? 0}% · outlining and measuring the tumor. You can leave this page; the case stays in the worklist.
      </p>
    </section>
  );

  const header = (
    <header className="flex flex-wrap items-center gap-x-3 gap-y-1 border-b bg-background px-3 py-2.5 sm:px-4">
      <Link href="/" title="Back to the worklist"
            className="flex size-9 items-center justify-center rounded-lg text-muted-foreground hover:bg-muted hover:text-foreground">
        <ArrowLeft className="size-5" aria-hidden />
        <span className="sr-only">Worklist</span>
      </Link>
      <h1 className="font-mono text-lg font-bold tracking-tight">{kase.code}</h1>
      <span className={`rounded-full px-2.5 py-1 text-xs font-semibold tabular-nums ${pill}`}>{label}</span>
      <span className="ml-auto text-sm text-muted-foreground">
        {scan ? (scan.file_format === "image" ? "2D slice" : "3D MRI") + " · " : ""}Created {whenOf(kase.created_at)} · no patient name is stored
      </span>
    </header>
  );

  // Nothing to look at yet (no scan, or a 2D case still running): a plain page with the upload and progress
  if (!image) {
    return (
      <>
        {header}
        <div className="min-h-0 flex-1 overflow-y-auto px-4 py-6 sm:px-8">
          <div className="mx-auto flex max-w-3xl flex-col gap-5">
            {alerts}
            {canUpload && <UploadForm caseId={kase.id} token={token} onUploaded={() => setVersion((v) => v + 1)} />}
            {kase.status === "new" && role !== "doctor" && <p className="text-muted-foreground">No scan uploaded yet.</p>}
            {progress && <div className="rounded-xl border bg-background p-5">{progress}</div>}
            {done && <Findings kase={kase} result={result} token={token} gradcam={false} />}
            {done && <Decision kase={kase} result={result} review={review} role={role} token={token} onChanged={() => setVersion((v) => v + 1)} />}
          </div>
        </div>
      </>
    );
  }

  // Reading room: the image fills the screen, findings and the decision live in a side panel that scrolls on its own
  return (
    <>
      {header}
      <div className="grid min-h-0 flex-1 lg:grid-cols-[minmax(0,1fr)_420px]">
        <div className="h-[75vh] min-h-0 bg-zinc-950 lg:h-auto">{image}</div>
        <aside aria-label="Case details" className="flex min-h-0 flex-col border-l bg-background">
          {done && (
            <div role="tablist" aria-label="Case details" className="flex gap-1 border-b px-3">
              {TABS.map(([t, text]) => (
                <button key={t} type="button" role="tab" aria-selected={tab === t} onClick={() => setTab(t)}
                        className={`min-h-11 border-b-2 px-3 text-sm font-semibold ${
                          tab === t ? "border-foreground text-foreground" : "border-transparent text-muted-foreground hover:text-foreground"}`}>
                  {text}
                </button>
              ))}
            </div>
          )}
          <div className="flex min-h-0 flex-1 flex-col gap-5 overflow-y-auto p-4">
            {alerts}
            {progress}
            {canUpload && <UploadForm caseId={kase.id} token={token} onUploaded={() => setVersion((v) => v + 1)} />}
            {done && (tab === "findings"
              ? <Findings kase={kase} result={result} token={token} gradcam={!gradcamMain} />
              : <Decision kase={kase} result={result} review={review} role={role} token={token} onChanged={() => setVersion((v) => v + 1)} />)}
            {done && tab === "findings" && (
              <button type="button" onClick={() => setTab("decision")}
                      className="min-h-11 rounded-lg bg-foreground px-4 font-semibold text-background hover:bg-foreground/90">
                {kase.status === "reviewed" ? "See the decision" : "Go to report and decision"}
              </button>
            )}
          </div>
          <p className="border-t px-4 py-2 text-xs text-muted-foreground">
            Research prototype, not a medical device. AI results are drafts until a doctor approves them.
          </p>
        </aside>
      </div>
    </>
  );
}

function Findings({ kase, result, token, gradcam }: { kase: CaseOut; result: ResultOut; token: string; gradcam: boolean }) {
  return <ResultSummary caseId={kase.id} result={result} reviewed={kase.status === "reviewed"} token={token} gradcam={gradcam} />;
}

function Decision({ kase, result, review, role, token, onChanged }: {
  kase: CaseOut;
  result: ResultOut;
  review: ReviewOut | null;
  role: string;
  token: string;
  onChanged: () => void;
}): ReactNode {
  return (
    <ReviewPanel
      key={`${kase.status}-${review?.id}`}  // a fresh form after each decision or reopen
      caseId={kase.id}
      caseCode={kase.code}
      result={result}
      review={review}
      reviewed={kase.status === "reviewed"}
      canDecide={role === "doctor"}
      token={token}
      onChanged={onChanged}
    />
  );
}
