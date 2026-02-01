"""Sample report drafts for the demo: the template and the LLM wording side by side, for typical cases.

    python backend/scripts/sample_reports.py   # from the repo root; reads the LLM settings from .env

The cases are made-up measurement sets (no patients, no images), chosen to show each path of the template:
a clear tumor, a midline-crossing one, a small lesion where the models disagree, no tumor, a 2D slice and a
doctor-corrected case. Writes docs/sample_reports.md.
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config import get_settings  # noqa: E402
from app.llm import LLM, LLMError  # noqa: E402
from app.report import Features, as_text, template_report  # noqa: E402

OUT = Path("docs/sample_reports.md")


def vols(wt: float, enhancing: float, necrosis: float, edema: float) -> dict[str, float]:
    return {"whole_tumor": wt, "tumor_core": round(enhancing + necrosis, 2), "enhancing": enhancing,
            "necrosis_non_enhancing": necrosis, "edema": edema}


CASES = {
    "Glioma, right parietal": Features("3d", "glioma", 0.94, None, vols(89.3, 21.6, 12.6, 55.1), False, "right", "parietal"),
    "Large glioma crossing the midline": Features("3d", "glioma", 0.88, None, vols(146.2, 38.4, 29.9, 77.9), False,
                                                  "both sides", "frontal"),
    "Small lesion, models disagree": Features("3d", "notumor", 0.71, None, vols(3.2, 0.6, 0.5, 2.1), False, "left",
                                              "temporal"),
    "No tumor": Features("3d", "notumor", 0.97, None, vols(0.0, 0.0, 0.0, 0.0), False, None, None),
    "2D slice, pituitary": Features("2d", "pituitary", 0.81, None, None, False, None, None),
    "Doctor changed the type and fixed the mask": Features("3d", "glioma", 0.62, "meningioma", vols(24.8, 9.1, 1.2, 14.5),
                                                           True, "left", "frontal"),
}


def main() -> None:
    s = get_settings()
    llm = LLM(s.llm_base_url, s.llm_api_key, s.llm_model, s.llm_timeout_s) if s.report_provider == "llm" else None
    lines = ["# Sample report drafts", "",
             "Made-up measurement sets (no patients, no images) showing the template and the LLM wording side by side.",
             f"LLM: {s.llm_model if llm else 'not set up'}. The LLM sees only the numbers and the template draft; "
             "its answer is refused (template used) if it changes a number, a tumor type or the sentence count.", ""]
    for name, f in CASES.items():
        draft = template_report(f)
        lines += [f"## {name}", "", "**Template**", "", "```", as_text(draft), "```", ""]
        if llm:
            t = time.time()
            try:
                out, status = llm.rewrite(f, draft), "passed the checks"
            except LLMError as e:
                out, status = None, f"refused, template used: {e}"
            lines += [f"**LLM** ({s.llm_model}, {time.time() - t:.1f} s, {status})", ""]
            if out:
                lines += ["```", as_text(out), "```", ""]
        print(f"{name}: done")
    OUT.write_text("\n".join(lines))
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
