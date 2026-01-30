"use client";

import { useEffect, useState } from "react";

import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";
import { api, type ReportOut } from "@/lib/api";

/** The report draft the doctor edits. The parent owns the text, because approving sends it as the final report. */
export function ReportEditor({ path, token, text, onText }: {
  path: string;  // /cases/{id}/results/{rid}/report
  token: string;
  text: string | null;  // null until the first draft is loaded
  onText: (text: string) => void;
}) {
  const [draft, setDraft] = useState<ReportOut | null>(null);
  const [busy, setBusy] = useState(false);
  const [note, setNote] = useState<string | null>(null);

  useEffect(() => {
    let live = true;
    api<ReportOut>(path, token).then(
      (d) => {
        if (!live) return;
        setDraft(d);
        onText(d.text);
      },
      (e: Error) => live && setNote(`The draft could not be loaded: ${e.message}`),
    );
    return () => {
      live = false;
    };
    // load once per result; onText is the parent's setter
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [path, token]);

  const edited = draft !== null && text !== null && text !== draft.text;

  async function draftWithAi() {
    if (edited && !window.confirm("Replace your edits with a new AI draft?")) return;
    setBusy(true);
    setNote(null);
    try {
      const d = await api<ReportOut>(`${path}/draft`, token, { method: "POST" });
      setDraft(d);
      onText(d.text);
      // The API falls back to the template on any LLM problem; say so instead of failing silently
      if (d.fallback) setNote(`The AI wording was not used: ${d.fallback}. Showing the template draft.`);
    } catch (e) {
      setNote(e instanceof Error ? e.message : "The AI draft failed.");
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="flex flex-col gap-2">
      <div className="flex flex-wrap items-center justify-between gap-2">
        <span className="text-sm font-medium">
          Report{" "}
          <span className="font-normal text-muted-foreground">
            {draft === null ? "· loading the draft…" : draft.provider === "llm"
              ? `· AI wording (${draft.model}) from the measurements${edited ? ", edited by you" : ""}`
              : `· template from the measurements${edited ? ", edited by you" : ""}`}
          </span>
        </span>
        <div className="flex gap-2">
          {edited && (
            <Button variant="ghost" className="h-9" disabled={busy} onClick={() => onText(draft.text)}>Undo my edits</Button>
          )}
          <Button variant="outline" className="h-9" disabled={busy || draft === null} onClick={draftWithAi}>
            {busy ? "Writing…" : "Draft with AI"}
          </Button>
        </div>
      </div>
      <Textarea
        value={text ?? ""}
        onChange={(e) => onText(e.target.value)}
        disabled={text === null}
        maxLength={10000}
        rows={12}
        aria-label="Report text"
        className="font-mono text-[13px] leading-relaxed"
      />
      {note && <p role="status" className="text-sm text-amber-800">{note}</p>}
      <p className="text-sm text-muted-foreground">
        The AI sees only the numbers above, never the images. Only the text you approve becomes the final report.
      </p>
    </div>
  );
}
