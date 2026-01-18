"use client";

import Link from "next/link";
import { useParams } from "next/navigation";
import { useEffect, useState } from "react";

import { useSession } from "@/components/session";
import { ResultSummary } from "@/components/result-summary";
import { ScanViewer } from "@/components/scan-viewer";
import { Progress, UploadForm } from "@/components/upload-form";
import { ErrorNote, Skeleton } from "@/components/states";
import { api, type ApiError, type CaseOut, type ResultOut, type ScanOut } from "@/lib/api";
import { statusOf, whenOf } from "@/lib/cases";

const POLL_MS = 2000;

export default function CasePage() {
  const { id } = useParams<{ id: string }>();
  const { token, role } = useSession();
  const [kase, setCase] = useState<CaseOut | null>(null);
  const [result, setResult] = useState<ResultOut | null>(null);
  const [scans, setScans] = useState<ScanOut[]>([]);
  const [error, setError] = useState<ApiError | null>(null);
  const [offline, setOffline] = useState(false);  // a progress poll failed; it keeps trying
  const [version, setVersion] = useState(0);  // bump to read the case again (after upload, when the AI ends)

  useEffect(() => {
    let live = true;
    Promise.all([
      api<CaseOut>(`/cases/${id}`, token),
      api<ResultOut[]>(`/cases/${id}/results`, token),
      api<ScanOut[]>(`/cases/${id}/scans`, token),
    ]).then(
      ([c, results, s]) => {
        if (!live) return;
        setCase(c);
        setResult(results[0] ?? null);
        setScans(s);
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
      <>
        <BackLink />
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
      </>
    );
  }

  const [label, pill] = statusOf({ status: kase.status, progress: result?.progress ?? null, decision: null });
  const canUpload = role === "doctor" && (kase.status === "new" || kase.status === "failed");
  // The scan the latest result was made from (or the newest upload); only 3D scans open in the viewer
  const scan = scans.find((s) => s.id === result?.scan_id) ?? scans.toSorted((a, b) => b.created_at.localeCompare(a.created_at))[0];
  const showViewer = scan && scan.file_format !== "image" && kase.status !== "new";

  return (
    <>
      <BackLink />
      <header className="flex flex-wrap items-center justify-between gap-4">
        <div className="flex flex-col gap-1">
          <h1 className="font-mono text-3xl font-bold tracking-tight">{kase.code}</h1>
          <p className="text-muted-foreground">Created {whenOf(kase.created_at)} · no patient name is stored</p>
        </div>
        {kase.status !== "reviewed" && (
          <span className={`rounded-full px-3 py-1.5 text-sm font-semibold tabular-nums ${pill}`}>{label}</span>
        )}
      </header>

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

      {canUpload && <UploadForm caseId={kase.id} token={token} onUploaded={() => setVersion((v) => v + 1)} />}
      {kase.status === "new" && role !== "doctor" && <p className="text-muted-foreground">No scan uploaded yet.</p>}

      {working && (
        <section className="flex flex-col gap-3 rounded-xl border bg-background p-5 sm:p-6" role="status">
          <h2 className="text-lg font-semibold">AI is analyzing the scan</h2>
          <Progress value={(result?.progress ?? 0) / 100} label="AI progress" />
          <p className="text-sm text-muted-foreground">
            {result?.progress ?? 0}% · outlining and measuring the tumor. You can leave this page; the case stays in the worklist.
          </p>
        </section>
      )}

      {showViewer && <ScanViewer caseId={kase.id} scanId={scan.id} token={token}
                                  resultId={result?.has_mask && result.scan_id === scan.id ? result.id : null} />}

      {result && (kase.status === "ready" || kase.status === "reviewed") && (
        <ResultSummary caseId={kase.id} result={result} token={token} />
      )}
    </>
  );
}

function BackLink() {
  return (
    <Link href="/" className="w-fit text-sm font-medium text-blue-700 hover:text-blue-800">
      ← Worklist
    </Link>
  );
}
