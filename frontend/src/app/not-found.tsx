import Link from "next/link";

import { Logo } from "@/components/logo";
import { buttonVariants } from "@/components/ui/button";

export default function NotFound() {
  return (
    <main className="flex min-h-screen flex-col items-center justify-center gap-5 bg-muted/50 p-8 text-center">
      <Logo />
      <h1 className="text-2xl font-bold">Page not found</h1>
      <p className="text-muted-foreground">This address doesn&apos;t exist in NeuroLens.</p>
      <Link href="/" className={buttonVariants({ className: "h-11 px-5 font-semibold" })}>
        Go to the worklist
      </Link>
    </main>
  );
}
