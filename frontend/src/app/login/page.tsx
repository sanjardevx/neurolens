"use client";

import Image from "next/image";
import { useRouter } from "next/navigation";
import { useState, type FormEvent } from "react";

import { Logo } from "@/components/logo";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { supabase } from "@/lib/supabase";

const ROLES = ["doctor", "admin"];

export default function LoginPage() {
  const router = useRouter();
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  async function signIn(e: FormEvent<HTMLFormElement>) {
    e.preventDefault();
    if (!supabase) {
      setError("Login is not set up: add the Supabase URL and publishable key to .env.local.");
      return;
    }
    const form = new FormData(e.currentTarget);
    setBusy(true);
    setError(null);
    const { data, error } = await supabase.auth.signInWithPassword({
      email: String(form.get("email")),
      password: String(form.get("password")),
    });
    if (error) {
      setError(error.message);
    } else if (!ROLES.includes(data.user?.app_metadata?.role)) {
      await supabase.auth.signOut();
      setError("This account has no NeuroLens role yet. Ask your admin.");
    } else {
      router.replace("/");
    }
    setBusy(false);
  }

  return (
    <div className="flex min-h-screen flex-wrap bg-background">
      <section className="order-2 flex min-w-0 flex-[1_1_520px] flex-col justify-between gap-10 bg-muted p-8 sm:p-14 lg:order-1">
        <div className="hidden lg:block">
          <Logo />
        </div>
        <div className="flex flex-wrap items-center gap-10">
          <div className="flex min-w-0 flex-[1_1_300px] flex-col gap-5">
            <h1 className="text-4xl leading-tight font-bold tracking-tight sm:text-5xl">
              Brain MRI review,
              <br />
              measured for you.
            </h1>
            <p className="max-w-md text-lg text-muted-foreground">
              The AI outlines the tumor in 3D, measures it and suggests the type. You check it and decide.
            </p>
          </div>
          <figure className="flex flex-[0_1_280px] flex-col gap-2">
            <Image
              src="/login-scan.png"
              alt="Axial FLAIR slice with the AI mask: green edema around a red and yellow tumor"
              width={470}
              height={500}
              priority
              className="rounded-2xl bg-black shadow-xl"
            />
            <figcaption className="text-xs text-muted-foreground">AI mask on a public BraTS 2021 scan</figcaption>
          </figure>
        </div>
        <p className="text-sm text-muted-foreground">Research prototype, not a medical device.</p>
      </section>

      {/* Form first on phones, so signing in needs no scrolling */}
      <section className="order-1 flex min-w-0 flex-[1_1_420px] items-center justify-center p-8 sm:p-14 lg:order-2">
        <form onSubmit={signIn} className="flex w-full max-w-sm flex-col gap-5" aria-labelledby="signin-title">
          <Logo className="lg:hidden" />
          <div className="flex flex-col gap-1.5">
            <h2 id="signin-title" className="text-2xl font-bold tracking-tight">Doctor sign in</h2>
            <p className="text-muted-foreground">For doctors and admins of your clinic.</p>
          </div>
          <div className="flex flex-col gap-2">
            <Label htmlFor="email">Work email</Label>
            <Input id="email" name="email" type="email" autoComplete="username" required className="h-11 text-base" />
          </div>
          <div className="flex flex-col gap-2">
            <Label htmlFor="password">Password</Label>
            <Input id="password" name="password" type="password" autoComplete="current-password" required className="h-11 text-base" />
          </div>
          {error && (
            <p role="alert" className="rounded-lg bg-destructive/10 px-3 py-2 text-sm text-destructive">
              {error}
            </p>
          )}
          <Button type="submit" disabled={busy} className="h-12 text-base font-semibold">
            {busy ? "Signing in…" : "Sign in"}
          </Button>
          <p className="rounded-lg bg-muted px-3 py-2.5 text-sm text-muted-foreground">
            Accounts are made by your admin. Patients do not have accounts and never see AI output.
          </p>
        </form>
      </section>
    </div>
  );
}
