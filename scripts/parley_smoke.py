"""Step 0: check the Parley key, list the models the key can use, and make one judge call.

    python scripts/parley_smoke.py            # list models + one call with JUDGE_MODEL
    python scripts/parley_smoke.py --no-call  # list models only
"""

from __future__ import annotations

import argparse
import json
import urllib.request

import dspy

from biasjudge.data import statement
from biasjudge.judge import build_judge, load_settings, make_lm


def list_models(settings) -> list[str]:
    req = urllib.request.Request(
        settings.base_url.rstrip("/") + "/models", headers={"Authorization": f"Bearer {settings.api_key}"}
    )
    with urllib.request.urlopen(req, timeout=20) as r:
        body = json.load(r)
    data = body.get("data", body)
    return sorted(m["id"] if isinstance(m, dict) else str(m) for m in data)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-call", action="store_true")
    args = ap.parse_args()
    settings = load_settings()
    print(f"base_url: {settings.base_url}")
    models = list_models(settings)
    print(f"{len(models)} models available to this key:")
    for m in models:
        print("  ", m)
    if settings.model not in models:
        print(f"\nWARNING: JUDGE_MODEL={settings.model} is not in the list above. Set JUDGE_MODEL in .env.")
    if args.no_call:
        return
    lm = make_lm(settings)
    dspy.configure(lm=lm)
    judge = build_judge()
    vid = "health_status:poor>age:middle-aged (45-64)"
    out = judge(statement=statement(vid))
    print(f"\nJUDGE_MODEL={settings.model}")
    print(f"item: {statement(vid)}")
    print(f"-> harmful_rating={out.harmful_rating} realism={out.realism}")
    print("\nParley smoke test OK.")


if __name__ == "__main__":
    main()
