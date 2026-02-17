"use client";

import { ListChecks, LogOut, PanelLeftClose, PanelLeftOpen, Plus, ScrollText } from "lucide-react";
import Link from "next/link";
import { usePathname } from "next/navigation";
import { type ReactNode, useState } from "react";

import { Logo } from "@/components/logo";
import { useSession } from "@/components/session";
import { supabase } from "@/lib/supabase";

const NAV = [
  { href: "/", label: "Worklist", icon: ListChecks, roles: ["doctor", "admin"] },
  { href: "/cases/new", label: "New case", icon: Plus, roles: ["doctor"] },
  { href: "/audit", label: "Audit log", icon: ScrollText, roles: ["admin"] },
];

const PIN_KEY = "neurolens.sidebar.pinned";
const ROW = "flex h-11 w-full items-center gap-3 rounded-lg px-3 text-[15px]";

export function AppShell({ children }: { children: ReactNode }) {
  const { email, role } = useSession();
  const path = usePathname();
  // A case page is a full-screen workspace (viewer + side panel, no page scroll); other pages are documents
  const workspace = /^\/cases\/(?!new$)[^/]+$/.test(path);

  // Slim icon rail by default, so the image gets the width. Hover (or keyboard focus) opens it over the page;
  // the pin keeps it open and pushes the page over. The pin is remembered per browser.
  // The shell renders only after the client-side session check, so storage can be read on the first render
  const [pinned, setPinned] = useState(() => {
    try {
      return localStorage.getItem(PIN_KEY) === "1";
    } catch {
      return false;  // storage can be blocked; the rail still works
    }
  });
  const [peek, setPeek] = useState(false);
  function togglePin() {
    const next = !pinned;
    setPinned(next);
    setPeek(false);
    try {
      localStorage.setItem(PIN_KEY, next ? "1" : "0");
    } catch {}
  }
  const open = pinned || peek;

  return (
    <div className="flex min-h-screen bg-muted/50">
      <div className={`sticky top-0 z-30 h-screen shrink-0 transition-[width] duration-200 motion-reduce:transition-none ${pinned ? "w-56" : "w-16"}`}>
        <nav
          aria-label="Main"
          onMouseEnter={() => setPeek(true)}
          onMouseLeave={() => setPeek(false)}
          onFocus={() => setPeek(true)}
          onBlur={(e) => !e.currentTarget.contains(e.relatedTarget) && setPeek(false)}
          className={`absolute inset-y-0 left-0 flex flex-col gap-4 overflow-hidden border-r bg-background px-2.5 py-3 transition-[width,box-shadow] duration-200 motion-reduce:transition-none ${
            open ? "w-56" : "w-16"} ${peek && !pinned ? "shadow-xl" : ""}`}
        >
          <div className="flex h-11 items-center justify-between">
            <Link href="/" title="NeuroLens" className="flex h-11 shrink-0 items-center px-1.5">
              <Logo iconOnly={!open} />
            </Link>
            {open && (
              <button
                type="button"
                onClick={togglePin}
                title={pinned ? "Close sidebar" : "Keep sidebar open"}
                aria-pressed={pinned}
                className="flex size-8 shrink-0 items-center justify-center rounded-md text-muted-foreground hover:bg-muted hover:text-foreground"
              >
                {pinned ? <PanelLeftClose className="size-4" aria-hidden /> : <PanelLeftOpen className="size-4" aria-hidden />}
                <span className="sr-only">{pinned ? "Close sidebar" : "Keep sidebar open"}</span>
              </button>
            )}
          </div>
          {/* Touch screens have no hover: the first tap on the rail opens it */}
          {!open && (
            <button type="button" onClick={togglePin} title="Open sidebar"
                    className="flex size-11 items-center justify-center rounded-lg text-muted-foreground hover:bg-muted hover:text-foreground">
              <PanelLeftOpen className="size-5" aria-hidden />
              <span className="sr-only">Open sidebar</span>
            </button>
          )}
          <ul className="flex flex-col gap-1">
            {NAV.filter((item) => item.roles.includes(role)).map(({ href, label, icon: Icon }) => {
              const active = href === "/" ? path === "/" : path.startsWith(href);
              return (
                <li key={href}>
                  <Link
                    href={href}
                    title={open ? undefined : label}
                    aria-current={active ? "page" : undefined}
                    className={`${ROW} ${active ? "bg-muted font-semibold text-foreground" : "text-muted-foreground hover:bg-muted hover:text-foreground"}`}
                  >
                    <Icon className="size-5 shrink-0" aria-hidden />
                    <span className={`whitespace-nowrap ${open ? "" : "sr-only"}`}>{label}</span>
                  </Link>
                </li>
              );
            })}
          </ul>
          <div className="mt-auto flex flex-col gap-1">
            <div className="flex items-center gap-3 px-1.5 py-1" title={open ? undefined : `${email} · ${role}`}>
              <span className="flex size-8 shrink-0 items-center justify-center rounded-full bg-muted text-sm font-semibold uppercase" aria-hidden>
                {email.slice(0, 1)}
              </span>
              <span className={`flex min-w-0 flex-col text-sm leading-tight ${open ? "" : "sr-only"}`}>
                <span className="truncate font-medium">{email}</span>
                <span className="capitalize text-muted-foreground">{role}</span>
              </span>
            </div>
            <button type="button" title={open ? undefined : "Sign out"} onClick={() => supabase?.auth.signOut()}
                    className={`${ROW} text-muted-foreground hover:bg-muted hover:text-foreground`}>
              <LogOut className="size-5 shrink-0" aria-hidden />
              <span className={`whitespace-nowrap ${open ? "" : "sr-only"}`}>Sign out</span>
            </button>
          </div>
        </nav>
      </div>
      {workspace ? (
        <main className="flex min-w-0 flex-1 flex-col lg:h-screen lg:overflow-hidden">{children}</main>
      ) : (
        <main className="mx-auto flex w-full min-w-0 max-w-6xl flex-1 flex-col gap-6 px-4 py-9 sm:px-10">
          {children}
          <p className="mt-auto pt-6 text-sm text-muted-foreground">
            Research prototype, not a medical device. AI results are drafts until a doctor approves them.
          </p>
        </main>
      )}
    </div>
  );
}
