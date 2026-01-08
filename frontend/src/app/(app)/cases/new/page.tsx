"use client";

import { useRouter } from "next/navigation";
import { useEffect, useRef, useState } from "react";

import { useSession } from "@/components/session";
import { api, type CaseOut } from "@/lib/api";

/** Makes a new case right away, so its anonymous code can be shown on the upload screen. */
export default function NewCasePage() {
  const { token } = useSession();
  const router = useRouter();
  const [error, setError] = useState<string | null>(null);
  const started = useRef(false);  // React runs effects twice in dev; make only one case

  useEffect(() => {
    if (started.current) return;
    started.current = true;
    api<CaseOut>("/cases", token, { method: "POST" }).then(
      (c) => router.replace(`/cases/${c.id}`),
      (e: Error) => setError(e.message),
    );
  }, [token, router]);

  return error ? (
    <p role="alert" className="rounded-lg bg-destructive/10 px-3 py-2 text-sm text-destructive">{error}</p>
  ) : (
    <p role="status" className="text-muted-foreground">Creating a new case…</p>
  );
}
