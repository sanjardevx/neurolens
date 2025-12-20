# NeuroLens API

FastAPI service for doctors: login (Supabase Auth), cases, scans, AI results, review and an audit log.
Doctor-only: patients have no accounts and never see AI output.

```bash
pip install -e "backend[dev]"
pytest backend
cd backend && uvicorn app.main:app --reload   # needs SUPABASE_URL and DATABASE_URL in .env
```

## Roles

Only two roles: `doctor` and `admin`. The role is read from the login token's `app_metadata.role`,
which users cannot change themselves. Set it once per account in the Supabase SQL editor:

```sql
update auth.users
set raw_app_meta_data = raw_app_meta_data || '{"role": "doctor"}'::jsonb   -- or "admin"
where email = 'doctor@example.com';
```

The new role is in the token after the next login.

| | doctor | admin |
|---|---|---|
| Create a case | yes | no |
| See cases | own only (others' return 404) | all |
| Read the audit log | no | yes |

## Data safety

- Tables: `cases`, `scans`, `results`, `reviews`, `audit_log`. A case has an anonymous code (`NL-7KQ2XD`), never a name.
- On Supabase, row-level security is switched on with no policies, so the public REST API sees nothing;
  only this API (database owner connection) reads and writes.
- Creating and opening a case is written to `audit_log`.
