import { TUMOR_LABELS, type CaseRow } from "@/lib/api";

const PILL = {
  review: "bg-amber-100 text-amber-900",
  working: "bg-blue-50 text-blue-700",
  approved: "bg-green-100 text-green-900",
  rejected: "bg-red-100 text-red-900",
  muted: "bg-muted text-muted-foreground",
};

export function statusOf(c: Pick<CaseRow, "status" | "progress" | "decision">): [string, string] {
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

export function resultOf(c: Pick<CaseRow, "tumor_type" | "tumor_cm3" | "confidence">): string {
  if (!c.tumor_type) return "—";
  const type = TUMOR_LABELS[c.tumor_type] ?? c.tumor_type;
  // The 3D model measures the tumor; a single 2D slice only gives the type and its confidence
  if (c.tumor_cm3) return `${type} · ${c.tumor_cm3.toFixed(1)} cm³`;
  return c.confidence !== null ? `${type} · ${Math.round(c.confidence * 100)}%` : type;
}

export function whenOf(iso: string): string {
  // SQLite (local dev) sends times without a zone; they are UTC
  const d = new Date(/[zZ]|[+-]\d\d:\d\d$/.test(iso) ? iso : iso + "Z");
  const days = Math.round((new Date().setHours(0, 0, 0, 0) - new Date(d).setHours(0, 0, 0, 0)) / 86_400_000);
  const time = d.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
  if (days === 0) return `Today ${time}`;
  if (days === 1) return `Yesterday ${time}`;
  if (days < 7) return d.toLocaleDateString([], { weekday: "long" });
  return d.toLocaleDateString([], { day: "numeric", month: "short", year: "numeric" });
}
