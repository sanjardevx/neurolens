import { Button } from "@/components/ui/button";

/** One look for every error: what went wrong in plain words, and a way to try again. */
export function ErrorNote({ message, onRetry }: { message: string; onRetry?: () => void }) {
  return (
    <div role="alert" className="flex flex-wrap items-center justify-between gap-3 rounded-lg bg-destructive/10 px-4 py-3 text-destructive">
      <span>{message}</span>
      {onRetry && (
        <Button variant="outline" className="h-9 border-destructive/30 bg-background" onClick={onRetry}>
          Try again
        </Button>
      )}
    </div>
  );
}

/** Gray placeholder lines while data loads; screen readers hear the label instead. */
export function Skeleton({ label, lines = 3 }: { label: string; lines?: number }) {
  return (
    <div role="status" className="flex flex-col gap-3">
      <span className="sr-only">{label}</span>
      {Array.from({ length: lines }, (_, i) => (
        <div key={i} className="h-12 animate-pulse rounded-lg bg-muted motion-reduce:animate-none" aria-hidden />
      ))}
    </div>
  );
}
