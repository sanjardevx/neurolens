# NeuroLens dashboard

Next.js (App Router) + Tailwind + shadcn/ui. Doctor-only: Supabase login, worklist, upload, case review.

```bash
cp .env.example .env.local   # Supabase URL + publishable key, API URL
npm install
npm run dev                  # http://localhost:3000
npm run lint && npm run build
```

- `src/app/login` — sign in with Supabase; accounts without a `doctor` / `admin` role are refused.
- `src/app/(app)` — pages behind the login (`SessionGate`), inside the app shell: a slim icon rail that opens on hover and can be pinned open.
- `src/app/(app)/page.tsx` — worklist: tabs by status, largest tumors first, refreshes while the AI works.
- `src/app/(app)/cases/new`, `cases/[id]` — a new case gets its anonymous code at once. The case page is a full-screen
  reading room: the scan fills the screen, a side panel holds the findings (`components/result-summary.tsx`) and the
  report and decision; only the panel scrolls.
- `components/scan-viewer.tsx` — NiiVue viewer: MRI type, plane, slice slider, AI mask per region, brush / eraser
  to fix the mask (saved next to the AI's mask), and a synced side-by-side compare of the AI mask with the fix.
  `components/review-panel.tsx` — approve / edit / reject with notes.
- `src/app/(app)/audit` — admin-only audit log: who did what to which case, filters by action and case code.
- `src/lib/api.ts` — API calls with the login token; an expired login signs out. Uploads use XHR for real progress.
- `src/lib/supabase.ts` — browser login client. The API checks the token and role on every call.
- Every page has loading, error (with "Try again") and empty states (`components/states.tsx`).

Colors: neutral (white, light gray, near-black) with one blue accent for focus and links; the scan viewer is dark.
