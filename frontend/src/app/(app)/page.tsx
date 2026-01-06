"use client";

import Link from "next/link";
import { useEffect, useState } from "react";

import { useSession } from "@/components/session";
import { buttonVariants } from "@/components/ui/button";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import { api, TUMOR_LABELS, type CaseRow } from "@/lib/api";

const POLL_MS = 5000;

type Tab = "review" | "processing" | "done" | "all";
const TABS: [Tab, string][] = [
  ["review", "Needs review"],
  ["processing", "Processing"],
  ["done", "Done"],
  ["all", "All"],
];

const TAB_OF: Record<CaseRow["status"], Tab> = {
  ready: "review",
  new: "processing",
  processing: "processing",
  reviewed: "done",
  failed: "done",
};

const PILL = {
  review: "bg-amber-100 text-amber-900",
  working: "bg-blue-50 text-blue-700",
  approved: "bg-green-100 text-green-900",
  rejected: "bg-red-100 text-red-900",
  muted: "bg-muted text-muted-foreground",
};

function statusOf(c: CaseRow): [string, string] {
  switch (c.status) {
    case "ready":
      return ["Needs review", PILL.review];
    case "new":
      return ["No scan yet", PILL.muted];
    case "processing":
      return [`AI working… ${c.progress ?? 0}%`, PILL.working];
    case "failed":
      return ["Failed", PILL.muted];
    case "reviewed":
      if (c.decision === "rejected") return ["Rejected", PILL.rejected];
      return [c.decision === "edited" ? "Edited" : "Approved", PILL.approved];
  }
}

function resultOf(c: CaseRow): string {
  if (!c.tumor_type) return "—";
  const type = TUMOR_LABELS[c.tumor_type] ?? c.tumor_type;
  // The 3D model measures the tumor; a single 2D slice only gives the type and its confidence
  if (c.tumor_cm3) return `${type} · ${c.tumor_cm3.toFixed(1)} cm³`;
  return c.confidence !== null ? `${type} · ${Math.round(c.confidence * 100)}%` : type;
}

function whenOf(iso: string): string {
  // SQLite (local dev) sends times without a zone; they are UTC
  const d = new Date(/[zZ]|[+-]\d\d:\d\d$/.test(iso) ? iso : iso + "Z");
  const days = Math.round((new Date().setHours(0, 0, 0, 0) - new Date(d).setHours(0, 0, 0, 0)) / 86_400_000);
  const time = d.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
  if (days === 0) return `Today ${time}`;
  if (days === 1) return `Yesterday ${time}`;
  if (days < 7) return d.toLocaleDateString([], { weekday: "long" });
  return d.toLocaleDateString([], { day: "numeric", month: "short", year: "numeric" });
}

/** Largest tumors first (most urgent to read), then newest. */
function byUrgency(a: CaseRow, b: CaseRow): number {
  return (b.tumor_cm3 ?? -1) - (a.tumor_cm3 ?? -1) || b.created_at.localeCompare(a.created_at);
}

export default function WorklistPage() {
  const { token, role } = useSession();
  const [cases, setCases] = useState<CaseRow[] | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [tab, setTab] = useState<Tab>("review");

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
  }, [token, working]);

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

        {error ? (
          <p role="alert" className="m-4 rounded-lg bg-destructive/10 px-3 py-2 text-sm text-destructive">
            {error}
          </p>
        ) : cases === null ? (
          <p role="status" className="px-5 py-10 text-muted-foreground">Loading cases…</p>
        ) : rows.length === 0 ? (
          <p className="px-5 py-10 text-muted-foreground">
            {cases.length === 0 ? "No cases yet." : "Nothing here."}
          </p>
        ) : (
          <Table className="text-[15px]">
            <TableHeader>
              <TableRow>
                <TableHead className="h-11 pl-5">Case</TableHead>
                <TableHead>AI result</TableHead>
                <TableHead className="pr-5">Status</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {rows.map((c) => {
                const [label, pill] = statusOf(c);
                return (
                  <TableRow key={c.id}>
                    <TableCell className="py-3.5 pl-5">
                      <div className="font-mono font-semibold">{c.code}</div>
                      <div className="text-sm text-muted-foreground">{whenOf(c.created_at)}</div>
                    </TableCell>
                    <TableCell>{resultOf(c)}</TableCell>
                    <TableCell className="pr-5">
                      <span
                        title={c.status === "failed" ? (c.error ?? undefined) : undefined}
                        className={`inline-block rounded-full px-2.5 py-1 text-[13px] font-semibold tabular-nums ${pill}`}
                      >
                        {label}
                      </span>
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
