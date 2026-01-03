"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import type { ReactNode } from "react";

import { Logo } from "@/components/logo";
import { useSession } from "@/components/session";
import { Button } from "@/components/ui/button";
import { supabase } from "@/lib/supabase";

const NAV = [
  { href: "/", label: "Worklist", roles: ["doctor", "admin"] },
  { href: "/cases/new", label: "New case", roles: ["doctor"] },
  { href: "/audit", label: "Audit log", roles: ["admin"] },
];

export function AppShell({ children }: { children: ReactNode }) {
  const { email, role } = useSession();
  const path = usePathname();

  return (
    <div className="flex min-h-screen flex-wrap bg-muted/50">
      <nav aria-label="Main" className="flex max-w-full flex-[1_1_220px] flex-col gap-7 border-r bg-background px-3.5 py-6">
        <Link href="/" className="px-2.5">
          <Logo />
        </Link>
        <ul className="flex flex-col gap-1">
          {NAV.filter((item) => item.roles.includes(role)).map((item) => {
            const active = item.href === "/" ? path === "/" : path.startsWith(item.href);
            return (
              <li key={item.href}>
                <Link
                  href={item.href}
                  aria-current={active ? "page" : undefined}
                  className={`flex min-h-11 items-center rounded-lg px-3 text-[15px] ${
                    active ? "bg-muted font-semibold" : "text-foreground/80 hover:bg-muted"
                  }`}
                >
                  {item.label}
                </Link>
              </li>
            );
          })}
        </ul>
        <div className="mt-auto flex flex-col gap-2 px-3 text-sm">
          <span className="truncate font-medium" title={email}>{email}</span>
          <span className="text-muted-foreground capitalize">{role}</span>
          <Button variant="outline" className="mt-2 h-10" onClick={() => supabase?.auth.signOut()}>
            Sign out
          </Button>
        </div>
      </nav>
      <main className="flex min-w-0 max-w-6xl flex-[999_1_560px] flex-col gap-6 px-4 py-9 sm:px-12">
        {children}
        <p className="mt-auto pt-6 text-sm text-muted-foreground">
          Research prototype, not a medical device. AI results are drafts until a doctor approves them.
        </p>
      </main>
    </div>
  );
}
