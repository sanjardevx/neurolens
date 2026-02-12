"use client";

import Link from "next/link";
import { useEffect, useState } from "react";

import { useSession } from "@/components/session";
import { buttonVariants } from "@/components/ui/button";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import { ErrorNote, Skeleton } from "@/components/states";
import { api, type CaseRow } from "@/lib/api";
import { modelOf, resultOf, statusOf, whenOf } from "@/lib/cases";

const POLL_MS = 5000;

type Tab = "review" | "processing" | "done" | "all";
const TABS: [Tab, string][] = [
  ["review", "Needs review"],
  ["processing", "Processing"],
  ["done", "Done"],
  ["all", "All"],
];

const EMPTY_TAB: Record<Tab, string> = {
  review: "Nothing is waiting for your review.",
  processing: "No scans are being analyzed.",
  done: "No reviewed cases yet.",
  all: "No cases.",
};

const TAB_OF: Record<CaseRow["status"], Tab> = {
  ready: "review",
  new: "processing",
  processing: "processing",
  reviewed: "done",
  failed: "done",
};

/** Largest tumors first (most urgent to read), then newest. */
function byUrgency(a: CaseRow, b: CaseRow): number {
  return (b.tumor_cm3 ?? -1) - (a.tumor_cm3 ?? -1) || b.created_at.localeCompare(a.created_at);
}

export default function WorklistPage() {
  const { token, role } = useSession();
  const [cases, setCases] = useState<CaseRow[] | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [tab, setTab] = useState<Tab>("review");
  const [attempt, setAttempt] = useState(0);  // bump to load again after an error

  // Refresh while the AI is working, so progress and finished results appear by themselves
  const working = cases?.some((c) => c.status === "processing") ?? false;
  useEffect(() => {
    let live = true;  // ignore answers that arrive after the page is left
    const load = () =>
      api<CaseRow[]>("/cases", token).then(
        (rows) => {
          if (!live) return;
          setCases(rows);
          setError(null);
        },
        (e: Error) => live && setError(e.message),
      );
    load();
    const id = working ? setInterval(load, POLL_MS) : undefined;
    return () => {
      live = false;
      clearInterval(id);
    };
  }, [token, working, attempt]);

  const count = (t: Tab) => cases?.filter((c) => t === "all" || TAB_OF[c.status] === t).length ?? 0;
  const rows = (cases ?? []).filter((c) => tab === "all" || TAB_OF[c.status] === tab).sort(byUrgency);

  return (
    <>
      <header className="flex flex-wrap items-center justify-between gap-4">
        <h1 className="text-3xl font-bold tracking-tight">Worklist</h1>
        {role === "doctor" && (
          <Link href="/cases/new" className={buttonVariants({ className: "h-11 px-5 text-[15px] font-semibold" })}>
            New case
          </Link>
        )}
      </header>

      <section className="rounded-xl border bg-background">
        <div role="group" aria-label="Filter cases" className="flex flex-wrap gap-1 border-b px-3">
          {TABS.map(([id, label]) => (
            <button
              key={id}
              type="button"
              aria-pressed={tab === id}
              onClick={() => setTab(id)}
              className={`-mb-px min-h-11 border-b-2 px-3.5 text-[15px] ${
                tab === id ? "border-foreground font-semibold" : "border-transparent text-muted-foreground hover:text-foreground"
              }`}
            >
              {label} <span className="tabular-nums">({count(id)})</span>
            </button>
          ))}
        </div>

        {error && (
          <div className="m-4">
            {/* After a failed refresh the last list stays visible below the note */}
            <ErrorNote message={cases ? `${error} The list below may be out of date.` : error}
                       onRetry={() => setAttempt((n) => n + 1)} />
          </div>
        )}
        {cases === null ? (
          !error && <div className="p-4"><Skeleton label="Loading cases…" /></div>
        ) : cases.length === 0 ? (
          <div className="flex flex-col items-start gap-4 px-5 py-10">
            <p className="text-lg font-semibold">No cases yet</p>
            {role === "doctor" ? (
              <>
                <p className="text-muted-foreground">Create a case and upload a brain MRI. The AI result lands here.</p>
                <Link href="/cases/new" className={buttonVariants({ className: "h-11 px-5 font-semibold" })}>
                  Create the first case
                </Link>
              </>
            ) : (
              <p className="text-muted-foreground">Cases appear here once doctors upload scans.</p>
            )}
          </div>
        ) : rows.length === 0 ? (
          <p className="px-5 py-10 text-muted-foreground">{EMPTY_TAB[tab]}</p>
        ) : (
          <Table className="text-[15px]">
            <TableHeader>
              <TableRow>
                <TableHead className="h-11 pl-5">Case</TableHead>
                <TableHead>AI result</TableHead>
                <TableHead>Status</TableHead>
                <TableHead className="pr-5 text-right"><span className="sr-only">Open</span></TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {rows.map((c) => {
                const [label, pill] = statusOf(c);
                const action = c.status === "ready" ? "Review" : c.status === "new" ? "Upload" : "Open";
                return (
                  <TableRow key={c.id}>
                    <TableCell className="py-3.5 pl-5">
                      {/* The code links too: on a phone the button column can be off screen */}
                      <Link href={`/cases/${c.id}`} tabIndex={-1} className="font-mono font-semibold hover:underline">
                        {c.code}
                      </Link>
                      <div className="text-sm text-muted-foreground">{whenOf(c.created_at)}</div>
                    </TableCell>
                    <TableCell>
                      {resultOf(c)}
                      {modelOf(c.model_version) && (
                        <div className="font-mono text-xs text-muted-foreground">{modelOf(c.model_version)}</div>
                      )}
                    </TableCell>
                    <TableCell>
                      <span
                        title={c.status === "failed" ? (c.error ?? undefined) : undefined}
                        className={`inline-block rounded-full px-2.5 py-1 text-[13px] font-semibold tabular-nums ${pill}`}
                      >
                        {label}
                      </span>
                    </TableCell>
                    <TableCell className="pr-5 text-right">
                      <Link
                        href={`/cases/${c.id}`}
                        aria-label={`${action} case ${c.code}`}
                        className={buttonVariants({
                          variant: c.status === "ready" ? "default" : "outline",
                          className: "h-10 px-4 font-semibold",
                        })}
                      >
                        {action}
                      </Link>
                    </TableCell>
                  </TableRow>
                );
              })}
            </TableBody>
          </Table>
        )}
      </section>
      <p className="text-sm text-muted-foreground">Largest tumors first.</p>
    </>
  );
}
