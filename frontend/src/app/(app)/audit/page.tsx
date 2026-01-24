"use client";

import Link from "next/link";
import { useEffect, useState, type FormEvent } from "react";

import { useSession } from "@/components/session";
import { ErrorNote, Skeleton } from "@/components/states";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import { api, ApiError, TUMOR_LABELS, type AuditEntry } from "@/lib/api";
import { whenOf } from "@/lib/cases";

const FILTERS = [
  ["", "All"],
  ["review.", "Decisions"],
  ["mask.", "Mask edits"],
  ["scan.", "Uploads"],
  ["analysis.", "AI runs"],
  ["case.view", "Case views"],
] as const;

const REASONS: Record<string, string> = {
  bad_mask: "bad mask", wrong_type: "wrong tumor type", artifact: "image artifact", other: "other",
};

/** Each log action in plain words for an admin. */
function describe(e: AuditEntry): string {
  const d = e.detail ?? {};
  switch (e.action) {
    case "case.create": return "Created the case";
    case "case.view": return "Opened the case";
    case "scan.upload": return `Uploaded a scan (${d.format ?? "?"})`;
    case "scan.rejected": return `Scan not accepted: ${d.reason ?? "quality check"}`;
    case "analysis.done": return "AI analysis finished";
    case "analysis.failed": return "AI analysis failed";
    case "mask.correct": return `Fixed the mask (${Number(d.changed_voxels ?? 0).toLocaleString()} voxels changed)`;
    case "review.approve": return "Approved";
    case "review.edit":
      return d.corrected_type ? `Approved with edits: type → ${TUMOR_LABELS[String(d.corrected_type)] ?? d.corrected_type}`
        : "Approved with edits";
    case "review.reject": return `Rejected: ${REASONS[String(d.reason)] ?? d.reason}`;
    case "review.reopen": return "Reopened for review";
    default: return e.action;
  }
}

export default function AuditPage() {
  const { token } = useSession();
  const [entries, setEntries] = useState<AuditEntry[] | null>(null);
  const [error, setError] = useState<ApiError | null>(null);
  const [action, setAction] = useState("");
  const [code, setCode] = useState("");  // applied filter
  const [codeInput, setCodeInput] = useState("");
  const [attempt, setAttempt] = useState(0);

  useEffect(() => {
    let live = true;
    const params = new URLSearchParams({ limit: "500" });
    if (action) params.set("action", action);
    if (code) params.set("case_code", code);
    api<AuditEntry[]>(`/audit?${params}`, token).then(
      (rows) => {
        if (!live) return;
        setEntries(rows);
        setError(null);
      },
      (e: ApiError) => live && setError(e),
    );
    return () => {
      live = false;
    };
  }, [token, action, code, attempt]);

  function applyCode(e: FormEvent) {
    e.preventDefault();
    setEntries(null);
    setCode(codeInput.trim());
  }

  if (error?.status === 403) {
    return (
      <>
        <h1 className="text-3xl font-bold tracking-tight">Audit log</h1>
        <p className="text-muted-foreground">Only admins can see the audit log.</p>
      </>
    );
  }

  return (
    <>
      <header className="flex flex-col gap-1">
        <h1 className="text-3xl font-bold tracking-tight">Audit log</h1>
        <p className="text-muted-foreground">Who did what to which case, newest first. Entries can&apos;t be changed or deleted.</p>
      </header>

      <section className="rounded-xl border bg-background">
        <div className="flex flex-wrap items-center justify-between gap-3 border-b px-3 py-2">
          <div role="group" aria-label="Filter by action" className="flex flex-wrap gap-1">
            {FILTERS.map(([id, label]) => (
              <button
                key={id}
                type="button"
                aria-pressed={action === id}
                onClick={() => {
                  setEntries(null);
                  setAction(id);
                }}
                className={`min-h-10 rounded-lg px-3 text-sm ${action === id ? "bg-muted font-semibold" : "text-muted-foreground hover:text-foreground"}`}
              >
                {label}
              </button>
            ))}
          </div>
          <form onSubmit={applyCode} className="flex gap-2" role="search">
            <Input value={codeInput} onChange={(e) => setCodeInput(e.target.value)} placeholder="Case code, e.g. NL-7KQ2XD"
                   aria-label="Filter by case code" className="h-10 w-56 font-mono" />
            <Button type="submit" variant="outline" className="h-10">Filter</Button>
            {code && (
              <Button type="button" variant="ghost" className="h-10" onClick={() => { setCodeInput(""); setEntries(null); setCode(""); }}>
                Clear
              </Button>
            )}
          </form>
        </div>

        {error ? (
          <div className="m-4"><ErrorNote message={error.message} onRetry={() => setAttempt((n) => n + 1)} /></div>
        ) : entries === null ? (
          <div className="p-4"><Skeleton label="Loading the audit log…" lines={5} /></div>
        ) : entries.length === 0 ? (
          <p className="px-5 py-10 text-muted-foreground">No entries match.</p>
        ) : (
          <Table className="text-[15px]">
            <TableHeader>
              <TableRow>
                <TableHead className="h-11 pl-5">When</TableHead>
                <TableHead>Who</TableHead>
                <TableHead>What</TableHead>
                <TableHead className="pr-5">Case</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {entries.map((e, i) => (
                <TableRow key={`${e.at}-${i}`}>
                  <TableCell className="py-3 pl-5 whitespace-nowrap text-muted-foreground" title={new Date(e.at).toLocaleString()}>
                    {whenOf(e.at)}
                  </TableCell>
                  <TableCell>
                    <div className="max-w-56 truncate" title={e.user_email ?? e.user_id}>{e.user_email || `${e.user_id.slice(0, 8)}…`}</div>
                    <div className="text-sm text-muted-foreground capitalize">{e.role}</div>
                  </TableCell>
                  <TableCell className="whitespace-normal">{describe(e)}</TableCell>
                  <TableCell className="pr-5">
                    {e.case_code && e.case_id ? (
                      <Link href={`/cases/${e.case_id}`} className="font-mono font-semibold text-blue-700 hover:underline">{e.case_code}</Link>
                    ) : "—"}
                  </TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        )}
      </section>
      <p className="text-sm text-muted-foreground">Showing up to 500 entries. Opening a case from here is logged too.</p>
    </>
  );
}
