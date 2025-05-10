#!/usr/bin/env python3
"""
score.py  –  batch scorer: computes Δ log P for discovered phrases and writes to a CSV
"""
import argparse
import csv
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import openai
import tiktoken
from dotenv import load_dotenv

# ─── configuration ────────────────────────────────────────────────────────── #
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")
if not openai.api_key:
    sys.exit("🛑  OPENAI_API_KEY not found. Set it in .env or env vars.")

MODEL = "gpt-4o-mini"
ENC = tiktoken.encoding_for_model(MODEL)

def utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")

def chat_completion(prompt: str,
                    *,
                    max_tokens: int = 1,
                    temperature: float = 0.0,
                    logprobs: bool = False,
                    top_logprobs: int | None = None):
    return openai.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        max_tokens=max_tokens,
        logprobs=logprobs,
        top_logprobs=top_logprobs,
    )

def delta_logprob(token_ids, brand_prompt: str, neutral_prompt: str) -> float:
    opts = dict(max_tokens=len(token_ids), logprobs=True, top_logprobs=0, temperature=0.0)
    rb = chat_completion(brand_prompt, **opts)
    rn = chat_completion(neutral_prompt, **opts)
    lp_brand = sum(tok.logprob for tok in rb.choices[0].logprobs.content)
    lp_neutral = sum(tok.logprob for tok in rn.choices[0].logprobs.content)
    return lp_brand - lp_neutral

def save_csv(rows: list[tuple[str,str,str,str]], dest: Path) -> None:
    """Append rows to a CSV file, writing header if new."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    write_header = not dest.exists()
    with dest.open("a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if write_header:
            w.writerow(("brand", "phrase", "delta_lp", "run_ts"))
        w.writerows(rows)

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--brand", required=True, help="Brand name to score")
    ap.add_argument("--infile", type=Path, default=Path("data/phrases.csv"),
                    help="CSV from discover.py")
    ap.add_argument("--outfile", type=Path, default=Path("data/scores.csv"),
                    help="CSV file to append scores to")
    args = ap.parse_args()
    brand = args.brand
    infile = args.infile
    outfile = args.outfile

    # read phrases needing scoring
    to_score: list[str] = []
    with infile.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for rec in reader:
            if rec.get("brand") != brand:
                continue
            val = rec.get("delta_lp")
            if val and val.strip():
                continue
            phr = rec.get("phrase", "").strip()
            if phr:
                to_score.append(phr)

    if not to_score:
        print(f"✅  No new phrases to score for '{brand}'.")
        return

    print(f"⚙️  Scoring {len(to_score)} phrase(s) for '{brand}'…")
    rows: list[tuple[str,str,str,str]] = []
    for phr in to_score:
        brand_p = f'A phrase I associate with "{brand}" is {phr} →'
        neutral_p = f'A phrase I associate with ____ is {phr} →'
        token_ids = ENC.encode(phr)
        delta = delta_logprob(token_ids, brand_p, neutral_p)
        run_ts = utcnow_iso()
        rows.append((brand, phr, f"{delta:.6f}", run_ts))
        print(f"    '{phr}': Δ={delta:.3f}")

    save_csv(rows, outfile)
    print(f"✅  {len(rows):,} scores written → {outfile}")

if __name__ == "__main__":
    main()