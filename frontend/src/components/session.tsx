"use client";

import type { Session } from "@supabase/supabase-js";
import { useRouter } from "next/navigation";
import { createContext, useContext, useEffect, useState, type ReactNode } from "react";

import { supabase } from "@/lib/supabase";

export type Role = "doctor" | "admin";
export type SessionInfo = { email: string; role: Role; token: string };

const SessionContext = createContext<SessionInfo | null>(null);

function toInfo(session: Session | null): SessionInfo | null {
  const role = session?.user.app_metadata?.role;
  if (!session || (role !== "doctor" && role !== "admin")) return null;
  return { email: session.user.email ?? "", role, token: session.access_token };
}

/** Shows the app only to a logged-in doctor or admin. The API checks the token again on every call. */
export function SessionGate({ children }: { children: ReactNode }) {
  const router = useRouter();
  const [info, setInfo] = useState<SessionInfo | null>(null);

  useEffect(() => {
    if (!supabase) {
      router.replace("/login");
      return;
    }
    supabase.auth.getSession().then(({ data }) => {
      const next = toInfo(data.session);
      if (next) setInfo(next);
      else router.replace("/login");
    });
    const { data } = supabase.auth.onAuthStateChange((_event, session) => {
      const next = toInfo(session);
      setInfo(next);  // token refreshes land here too
      if (!next) router.replace("/login");
    });
    return () => data.subscription.unsubscribe();
  }, [router]);

  if (!info) {
    return <p className="m-auto text-muted-foreground" role="status">Loading…</p>;
  }
  return <SessionContext value={info}>{children}</SessionContext>;
}

export function useSession(): SessionInfo {
  const info = useContext(SessionContext);
  if (!info) throw new Error("useSession must be used inside SessionGate");
  return info;
}
