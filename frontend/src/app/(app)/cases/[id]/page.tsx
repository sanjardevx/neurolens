"use client";

import Link from "next/link";
import { useParams } from "next/navigation";
import { useEffect, useState } from "react";

import { useSession } from "@/components/session";
import { ResultSummary } from "@/components/result-summary";
import { Progress, UploadForm } from "@/components/upload-form";
import { api, type CaseOut, type ResultOut } from "@/lib/api";
import { statusOf, whenOf } from "@/lib/cases";

const POLL_MS = 2000;

export default function CasePage() {
  const { id } = useParams<{ id: string }>();
  const { token, role } = useSession();
  const [kase, setCase] = useState<CaseOut | null>(null);
  const [result, setResult] = useState<ResultOut | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [version, setVersion] = useState(0);  // bump to read the case again (after upload, when the AI ends)

  useEffect(() => {
    let live = true;
    Promise.all([api<CaseOut>(`/cases/${id}`, token), api<ResultOut[]>(`/cases/${id}/results`, token)]).then(
      ([c, results]) => {
        if (!live) return;
        setCase(c);
        setResult(results[0] ?? null);
      },
      (e: Error) => live && setError(e.message),
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
      api<ResultOut[]>(`/cases/${id}/results`, token).then(([r]) => {
        if (!live || !r) return;
        setResult(r);
        if (r.progress === 100 || r.error) setVersion((v) => v + 1);
      });
    }, POLL_MS);
    return () => {
      live = false;
      clearInterval(timer);
    };
  }, [working, id, token]);

  if (error) {
    return (
      <>
        <BackLink />
        <p role="alert" className="rounded-lg bg-destructive/10 px-3 py-2 text-sm text-destructive">{error}</p>
      </>
    );
  }
  if (!kase) return <p role="status" className="text-muted-foreground">Loading case…</p>;

  const [label, pill] = statusOf({ status: kase.status, progress: result?.progress ?? null, decision: null });
  const canUpload = role === "doctor" && (kase.status === "new" || kase.status === "failed");

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
