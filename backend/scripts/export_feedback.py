"""Export the doctors' feedback as a versioned training set (see app/feedback.py for the groups).

    python backend/scripts/export_feedback.py --out feedback [--tag fb-v1] [--zip]

Run from the repo root next to .env (DATABASE_URL, STORAGE_DIR). Only de-identified stored scans and case codes
are written; no names, emails or user ids. --zip also writes <tag>.zip for a Kaggle dataset upload.
"""
import argparse
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config import get_settings  # noqa: E402
from app.db import Database  # noqa: E402
from app.feedback import export_feedback  # noqa: E402
from app.files import FileStore  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="feedback", help="folder for the export (git-ignored)")
    ap.add_argument("--tag", help="version tag; default fb-<date>-<n>c")
    ap.add_argument("--zip", action="store_true")
    args = ap.parse_args()
    s = get_settings()
    db = Database(s.database_url)
    with db.sessions() as session:
        files = FileStore(s.storage_dir, s.supabase_url, s.supabase_secret_key, s.storage_bucket)
        summary = export_feedback(session, files, Path(args.out), args.tag)
    print(summary)
    if args.zip:
        print("zip:", shutil.make_archive(str(Path(args.out) / summary["tag"]), "zip", Path(args.out), summary["tag"]))


if __name__ == "__main__":
    main()
