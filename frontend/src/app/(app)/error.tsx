"use client";

import { ErrorNote } from "@/components/states";

/** Last safety net: a bug in a page shows this instead of a blank screen; the sidebar stays usable. */
export default function AppError({ reset }: { error: Error; reset: () => void }) {
  return (
    <>
      <h1 className="text-2xl font-bold">Something went wrong</h1>
      <ErrorNote message="This page hit an unexpected error. Your data is safe on the server." onRetry={reset} />
    </>
  );
}
