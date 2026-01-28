"""One OpenAI-compatible client (Ollama Cloud, local Ollama or any LLM API the user owns) that rewords the
template report. It gets the numbers only, never the image, and its answer is checked: any new number,
a changed tumor type or a broken format means the template is used instead."""
import json
import re

import httpx

from .report import TYPES, Features

SYSTEM = """You edit AI-generated brain MRI report drafts for a radiologist.
Rewrite the draft so it reads like a concise, professional radiology report.
Rules:
- Use only the facts in the draft and the measurements. Never add findings, numbers, diagnoses,
  differential diagnoses, grades or treatment advice.
- Keep every number exactly as written, with its unit.
- Keep the words "estimate" and "not calibrated" where they appear, and keep every note.
- Same number of findings and notes as the draft, in the same order.
Answer with JSON only: {"findings": [string], "impression": string, "notes": [string]}"""

NUMBER = re.compile(r"\d+(?:\.\d+)?")


class LLMError(Exception):
    """The answer can't be used; the reason is shown as the fallback reason."""


class LLM:
    def __init__(self, base_url: str, api_key: str, model: str, timeout_s: float = 60,
                 transport: httpx.BaseTransport | None = None):
        self.model = model
        self.http = httpx.Client(base_url=base_url.rstrip("/"), timeout=timeout_s, transport=transport,
                                 headers={"Authorization": f"Bearer {api_key}"} if api_key else {})

    def rewrite(self, f: Features, draft: dict) -> dict:
        facts = {"measurements": {k: v for k, v in vars(f).items() if v is not None},
                 "draft": {k: draft[k] for k in ("findings", "impression", "notes")}}
        try:
            r = self.http.post("/chat/completions", json={
                "model": self.model, "temperature": 0.2, "response_format": {"type": "json_object"},
                "messages": [{"role": "system", "content": SYSTEM},
                             {"role": "user", "content": json.dumps(facts, ensure_ascii=False)}]})
            r.raise_for_status()
            text = r.json()["choices"][0]["message"]["content"]
        except (httpx.HTTPError, KeyError, IndexError, ValueError) as e:
            raise LLMError(f"LLM call failed ({type(e).__name__})") from e
        return check(parse(text), draft)


def parse(text: str) -> dict:
    # Some models wrap JSON in a code fence even when asked not to
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip())
    try:
        out = json.loads(text)
    except ValueError as e:
        raise LLMError("the answer was not JSON") from e
    if not (isinstance(out, dict) and isinstance(out.get("impression"), str)
            and all(isinstance(out.get(k), list) and all(isinstance(s, str) for s in out[k]) for k in ("findings", "notes"))):
        raise LLMError("the answer had the wrong shape")
    return {"findings": out["findings"], "impression": out["impression"], "notes": out["notes"]}


def check(out: dict, draft: dict) -> dict:
    """Guards against a fluent but wrong report: the doctor must be able to trust every number."""
    if len(out["findings"]) != len(draft["findings"]) or len(out["notes"]) != len(draft["notes"]):
        raise LLMError("the answer dropped or added sentences")
    said = " ".join([*out["findings"], out["impression"], *out["notes"]])
    allowed = set(NUMBER.findall(" ".join([*draft["findings"], draft["impression"], *draft["notes"]])))
    if new := set(NUMBER.findall(said)) - allowed:
        raise LLMError(f"the answer has numbers not in the draft: {sorted(new)}")
    for t in TYPES.values():  # a tumor type the draft never named must not appear
        if t != "no tumor" and t in said.lower() and t not in " ".join([*draft["findings"], draft["impression"]]).lower():
            raise LLMError(f"the answer names a tumor type not in the draft: {t}")
    return {**out, "disclaimer": draft["disclaimer"]}
