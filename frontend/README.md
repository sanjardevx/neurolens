# NeuroLens dashboard

Next.js (App Router) + Tailwind + shadcn/ui. Doctor-only: Supabase login, worklist, upload, case review.

```bash
cp .env.example .env.local   # Supabase URL + publishable key, API URL
npm install
npm run dev                  # http://localhost:3000
npm run lint && npm run build
```

- `src/app/login` — sign in with Supabase; accounts without a `doctor` / `admin` role are refused.
- `src/app/(app)` — pages behind the login (`SessionGate`), inside the app shell (sidebar, sign out).
- `src/app/(app)/page.tsx` — worklist: tabs by status, largest tumors first, refreshes while the AI works.
- `src/app/(app)/cases/new`, `cases/[id]` — a new case gets its anonymous code at once; the case page uploads the scan,
  follows the AI job and shows the result summary (`components/result-summary.tsx`).
- `components/scan-viewer.tsx` — NiiVue viewer: MRI type, plane, slice slider, AI mask per region, brush / eraser
  to fix the mask (saved next to the AI's mask). `components/review-panel.tsx` — approve / edit / reject with notes.
- `src/app/(app)/audit` — admin-only audit log: who did what to which case, filters by action and case code.
- `src/lib/api.ts` — API calls with the login token; an expired login signs out. Uploads use XHR for real progress.
- `src/lib/supabase.ts` — browser login client. The API checks the token and role on every call.
- Every page has loading, error (with "Try again") and empty states (`components/states.tsx`).

Colors: neutral (white, light gray, near-black) with one blue accent for focus and links; the scan viewer is dark.
