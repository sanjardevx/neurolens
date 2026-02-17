export function Logo({ className = "", iconOnly = false }: { className?: string; iconOnly?: boolean }) {
  return (
    <span className={`inline-flex items-center gap-2.5 ${className}`}>
      {iconOnly && <span className="sr-only">NeuroLens</span>}
      <svg width="28" height="28" viewBox="0 0 32 32" fill="none" aria-hidden="true">
        <circle cx="14" cy="14" r="10" stroke="currentColor" strokeWidth="2.4" />
        <path d="M8.5 14c1.6-3 3.2-3 4.6 0s3 3 4.6 0" stroke="currentColor" strokeWidth="2.2" strokeLinecap="round" />
        <path d="M21.5 21.5 28 28" stroke="currentColor" strokeWidth="2.6" strokeLinecap="round" />
      </svg>
      {!iconOnly && <span className="text-lg font-bold tracking-tight">NeuroLens</span>}
    </span>
  );
}
