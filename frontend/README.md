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
- `src/lib/supabase.ts` — browser login client. The API checks the token and role on every call.

Colors: neutral (white, light gray, near-black) with one blue accent for focus and links; the scan viewer is dark.
