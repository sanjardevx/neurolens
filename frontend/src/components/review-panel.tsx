"use client";

import { useState } from "react";

import { ReportEditor } from "@/components/report-editor";
import { ErrorNote } from "@/components/states";
import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";
import { api, downloadFile, TUMOR_LABELS, type ResultOut, type ReviewOut } from "@/lib/api";
import { whenOf } from "@/lib/cases";

const REASONS = [
  ["bad_mask", "Bad mask"],
  ["wrong_type", "Wrong tumor type"],
  ["artifact", "Image artifact"],
  ["other", "Other"],
] as const;
type Reason = (typeof REASONS)[number][0];

const DONE_TEXT = { approved: "Approved", edited: "Approved with your edits", rejected: "Rejected" };

/** The doctor's decision on the AI result: approve, edit (type and / or mask) or reject with a reason. */
export function ReviewPanel({ caseId, caseCode, result, review, reviewed, canDecide, token, onChanged }: {
  caseId: string;
  caseCode: string;
  result: ResultOut;
  review: ReviewOut | null;  // the latest decision, if any
  reviewed: boolean;
  canDecide: boolean;  // the case's doctor; admins only read
  token: string;
  onChanged: () => void;
}) {
  const [step, setStep] = useState<"open" | "edit" | "reject">("open");
  const [note, setNote] = useState("");
  const [type, setType] = useState(result.tumor_type ?? "");
  const [reason, setReason] = useState<Reason | null>(null);
  const [report, setReport] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const maskFixed = result.has_corrected_mask;

  async function send(path: string, body?: object) {
    setBusy(true);
    setError(null);
    try {
      await api(path, token, body
        ? { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) }
        : { method: "POST" });
      onChanged();
    } catch (e) {
      setError(e instanceof Error ? e.message : "The decision was not saved.");
    } finally {
      setBusy(false);
    }
  }

  async function downloadPdf() {
    setBusy(true);
    setError(null);
    try {
      await downloadFile(`/cases/${caseId}/report.pdf`, token, `${caseCode}_report.pdf`);
    } catch (e) {
      setError(e instanceof Error ? `The PDF could not be made: ${e.message}` : "The PDF could not be made.");
    } finally {
      setBusy(false);
    }
  }

  const decide = (decision: ReviewOut["decision"], extra: object = {}) =>
    send(`/cases/${caseId}/reviews`, { result_id: result.id, decision, note: note.trim() || null, ...extra });

  if (reviewed && review) {
    const why = REASONS.find(([r]) => r === review.reason)?.[1];
    return (
      <section className="flex flex-col gap-3" aria-label="Decision">
        <h2 className="font-semibold">
          {DONE_TEXT[review.decision]}{why ? `: ${why.toLowerCase()}` : ""}
          <span className="font-normal text-muted-foreground"> · {whenOf(review.created_at)}</span>
        </h2>
        {review.corrected_type && (
          <p>Tumor type changed to <strong>{TUMOR_LABELS[review.corrected_type] ?? review.corrected_type}</strong>.</p>
        )}
        {review.note && <p className="whitespace-pre-wrap rounded-lg bg-muted/60 px-4 py-3">{review.note}</p>}
        {review.report_text && (
          <div className="flex flex-col gap-1">
            <h3 className="text-sm font-medium">Final report</h3>
            <pre className="whitespace-pre-wrap rounded-lg border px-4 py-3 font-mono text-[13px] leading-relaxed">{review.report_text}</pre>
          </div>
        )}
        {error && <ErrorNote message={error} />}
        <div className="flex flex-wrap gap-2">
          {review.decision !== "rejected" && (
            <Button className="h-10" disabled={busy} onClick={downloadPdf}>Download PDF</Button>
          )}
          {canDecide && (
            <Button variant="outline" className="h-10" disabled={busy} onClick={() => send(`/cases/${caseId}/reopen`)}>
              Change decision
            </Button>
          )}
        </div>
      </section>
    );
  }

  if (!canDecide) {
    return <p className="text-muted-foreground">Waiting for the doctor&apos;s review.</p>;
  }

  return (
    <section className="flex flex-col gap-4" aria-labelledby="decision-title">
      <h2 id="decision-title" className="font-semibold">Your decision</h2>
      <label className="flex flex-col gap-2">
        <span className="text-sm font-medium">Your notes <span className="font-normal text-muted-foreground">(saved with the case)</span></span>
        <Textarea value={note} onChange={(e) => setNote(e.target.value)} maxLength={2000} rows={3}
                  placeholder="What you saw, what to follow up…" />
      </label>

      {step === "edit" && (
        <fieldset className="flex flex-col gap-2">
          <legend className="mb-2 text-sm font-medium">Tumor type</legend>
          <div className="flex flex-wrap gap-2">
            {Object.entries(TUMOR_LABELS).map(([id, label]) => (
              <label key={id} className={`flex min-h-10 cursor-pointer items-center gap-2 rounded-lg border px-3 ${type === id ? "border-foreground font-semibold" : ""}`}>
                <input type="radio" name="tumor-type" value={id} checked={type === id} onChange={() => setType(id)} className="accent-blue-600" />
                {label}{id === result.tumor_type && <span className="text-xs font-normal text-muted-foreground">(AI)</span>}
              </label>
            ))}
          </div>
          <p className="text-sm text-muted-foreground">
            To fix the outline, use &quot;Fix the mask&quot; in the viewer. If you change the type, update the report text too.
          </p>
        </fieldset>
      )}

      {step === "reject" && (
        <fieldset className="flex flex-col gap-2">
          <legend className="mb-2 text-sm font-medium">Why?</legend>
          <div className="flex flex-wrap gap-2">
            {REASONS.map(([id, label]) => (
              <label key={id} className={`flex min-h-10 cursor-pointer items-center gap-2 rounded-lg border px-3 ${reason === id ? "border-foreground font-semibold" : ""}`}>
                <input type="radio" name="reason" value={id} checked={reason === id} onChange={() => setReason(id)} className="accent-blue-600" />
                {label}
              </label>
            ))}
          </div>
          {reason === "other" && !note.trim() && <p className="text-sm text-muted-foreground">For &quot;Other&quot;, say why in your notes.</p>}
        </fieldset>
      )}

      {step !== "reject" && (
        <ReportEditor path={`/cases/${caseId}/results/${result.id}/report`} token={token} text={report} onText={setReport} />
      )}

      {error && <ErrorNote message={error} />}

      <div className="flex flex-wrap gap-2">
        {step === "open" && (
          <>
            {/* After a mask fix the AI was not right as it was, so the same button records an edit */}
            <Button className="h-11 px-6 font-semibold" disabled={busy} onClick={() => decide(maskFixed ? "edited" : "approved", { report_text: report?.trim() || null })}>
              {maskFixed ? "Approve with your mask edits" : "Approve"}
            </Button>
            <Button variant="outline" className="h-11" disabled={busy} onClick={() => setStep("edit")}>Edit type</Button>
            <Button variant="outline" className="h-11 text-destructive" disabled={busy} onClick={() => setStep("reject")}>Reject</Button>
          </>
        )}
        {step === "edit" && (
          <>
            <Button className="h-11 px-6 font-semibold"
                    disabled={busy || (type === result.tumor_type && !maskFixed)}
                    onClick={() => decide("edited", {
                      report_text: report?.trim() || null,
                      ...(type !== result.tumor_type ? { corrected_type: type } : {}),
                    })}>
              Save and approve
            </Button>
            <Button variant="ghost" className="h-11" disabled={busy} onClick={() => setStep("open")}>Cancel</Button>
          </>
        )}
        {step === "reject" && (
          <>
            <Button variant="destructive" className="h-11 px-6 font-semibold"
                    disabled={busy || !reason || (reason === "other" && !note.trim())}
                    onClick={() => decide("rejected", { reason })}>
              Reject
            </Button>
            <Button variant="ghost" className="h-11" disabled={busy} onClick={() => setStep("open")}>Cancel</Button>
          </>
        )}
      </div>
    </section>
  );
}
