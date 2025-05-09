#!/usr/bin/env python3
"""
discover.py  –  auto‑discovers words, short phrases, and competitors that GPT‑4o mini
associates with a single brand, plus any manual terms you supply.

USAGE
    # .env (same directory or project root)
    # OPENAI_API_KEY=sk‑....

    pip install openai python-dotenv tiktoken
    python discover.py --brand "McDonald's" \
                       --outfile data/phrases.csv \
                       --manual  data/manual_phrases.txt
"""

import argparse
import csv
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from dotenv import load_dotenv

import openai
import tiktoken

# ─── configuration ────────────────────────────────────────────────────────── #

load_dotenv()                           # ← read .env into os.environ
openai.api_key = os.getenv("OPENAI_API_KEY")
if not openai.api_key:
    sys.exit("🛑  OPENAI_API_KEY not found. Add it to .env or your env vars.")

MODEL               = "gpt-4o-mini"   # supports `logprobs`
TOP_K_TOKENS        = 20              # for top‑token scan
ITER_EXT_THRESHOLD  = 0.5             # stop growing phrase if Δ log P < this
HIGH_T_SAMPLES      = 100             # high‑temperature generations
TEMPERATURE_LIST    = 0.8
CSV_HEADER          = ("brand", "phrase", "source", "delta_lp", "created_at")

openai.api_key = os.getenv("OPENAI_API_KEY")
if not openai.api_key:
    sys.exit("Set the OPENAI_API_KEY environment variable.")

ENC = tiktoken.encoding_for_model(MODEL)

# ─── helpers ──────────────────────────────────────────────────────────────── #

def utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")

def save_csv(rows, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    write_header = not dest.exists()
    with dest.open("a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if write_header:
            w.writerow(CSV_HEADER)
        w.writerows(rows)

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

def prompt_templates(brand: str, kind: str) -> tuple[str, str]:
    quoted = f'"{brand}"'
    if kind == "word":
        return (f"A word I associate with {quoted} is →",
                "A word I associate with ____ is →")
    if kind == "competitor":
        return (f"A company that competes with {quoted} is →",
                "A company that competes with ____ is →")
    raise ValueError(f"Unknown kind: {kind}")

def delta_logprob(token_ids, brand_prompt, neutral_prompt) -> float:
    opts = dict(max_tokens=len(token_ids),
                logprobs=True, top_logprobs=0, temperature=0.0)

    rb = chat_completion(brand_prompt,   **opts)
    rn = chat_completion(neutral_prompt, **opts)

    lp_brand   = sum(tok.logprob for tok in rb.choices[0].logprobs.content)
    lp_neutral = sum(tok.logprob for tok in rn.choices[0].logprobs.content)
    return lp_brand - lp_neutral

# ─── discovery strategies ────────────────────────────────────────────────── #

def top_token_scan(brand: str) -> set[str]:
    brand_p, neutral_p = prompt_templates(brand, "word")
    print(f"🔍  [1/4] Top-token scan for '{brand}' (top {TOP_K_TOKENS} tokens)…")

    rb = chat_completion(brand_p,   logprobs=True, top_logprobs=TOP_K_TOKENS)
    rn = chat_completion(neutral_p, logprobs=True, top_logprobs=TOP_K_TOKENS)

    # each element in .content is the generated token; we want the top‑list
    brand_top = {item.token: item.logprob
                 for item in rb.choices[0].logprobs.content[0].top_logprobs}
    neut_top  = {item.token: item.logprob
                 for item in rn.choices[0].logprobs.content[0].top_logprobs}

    out = set()
    for tok, lp_b in brand_top.items():
        lp_n = neut_top.get(tok, -100.0)
        if lp_b - lp_n > 0:
            out.add(tok.lstrip("Ġ"))
    print(f"    → {len(out)} candidate tokens found")
    return out

def extend_phrase(brand: str, base_token: str) -> str:
    neutral_p = prompt_templates(brand, "word")[1]
    phrase = base_token
    while True:
        candidate_prompt = f'A phrase I associate with "{brand}" is {phrase} →'
        token_ids = ENC.encode(phrase)
        delta = delta_logprob(token_ids, candidate_prompt, neutral_p)
        if delta < ITER_EXT_THRESHOLD or len(token_ids) >= 6:
            return phrase
        r = chat_completion(candidate_prompt, max_tokens=1,
                            temperature=0.0, logprobs=True, top_logprobs=TOP_K_TOKENS)
        phrase = f"{phrase} {r.choices[0].message.content}".strip()

def iterative_extension(brand: str, base_tokens: set[str]) -> set[str]:
    total = len(base_tokens)
    print(f"🔍  [2/4] Iterative extension of {total} token(s)…")
    out = set()
    for idx, tok in enumerate(base_tokens, start=1):
        print(f"    [{idx}/{total}] Extending '{tok}'…")
        phrase = extend_phrase(brand, tok)
        if len(phrase.split()) > 1:
            out.add(phrase)
    print(f"    → {len(out)} extended phrase(s) generated")
    return out

def high_t_sampling(brand: str) -> set[str]:
    prompt = f'List five words or short phrases you associate with "{brand}".'
    print(f"🔍  [3/4] High-temperature sampling ({HIGH_T_SAMPLES} runs at T={TEMPERATURE_LIST})…")
    out = set()
    # sample multiple times and report progress
    report_every = max(1, HIGH_T_SAMPLES // 10)
    for i in range(HIGH_T_SAMPLES):
        # progress report
        if (i + 1) % report_every == 0 or i == HIGH_T_SAMPLES - 1:
            print(f"    {i+1}/{HIGH_T_SAMPLES} runs completed, current set size: {len(out)}")
        r = chat_completion(prompt, max_tokens=50, temperature=TEMPERATURE_LIST)
        text = r.choices[0].message.content
        for line in text.splitlines():
            cleaned = line.strip("•*- \t").lower()
            if 1 <= len(cleaned.split()) <= 4:
                out.add(cleaned)
    print(f"    → {len(out)} unique word/phrase(s) collected")
    return out

def competitor_sampling(brand: str) -> set[str]:
    prompt = f'List five companies that compete with "{brand}".'
    print(f"🔍  [4/4] Competitor sampling ({HIGH_T_SAMPLES} runs)…")
    out = set()
    report_every = max(1, HIGH_T_SAMPLES // 10)
    for i in range(HIGH_T_SAMPLES):
        if (i + 1) % report_every == 0 or i == HIGH_T_SAMPLES - 1:
            print(f"    {i+1}/{HIGH_T_SAMPLES} runs completed, competitors collected: {len(out)}")
        r = chat_completion(prompt, max_tokens=50, temperature=TEMPERATURE_LIST)
        text = r.choices[0].message.content
        for line in text.splitlines():
            cleaned = line.strip("•*- \t")
            if cleaned:
                out.add(cleaned)
    print(f"    → {len(out)} unique competitor(s) collected")
    return out

def manual_phrases(path: Path | None) -> set[str]:
    if not path or not path.exists():
        return set()
    with path.open(encoding="utf-8") as f:
        return {line.strip() for line in f if line.strip()}

# ─── CLI entry‑point ─────────────────────────────────────────────────────── #

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--brand", required=True, help='Brand name (wrap in quotes)')
    ap.add_argument("--outfile", type=Path, default=Path("data/phrases.csv"))
    ap.add_argument("--manual",  type=Path, help="txt file, one phrase per line")
    args = ap.parse_args()

    brand = args.brand
    manual_set = manual_phrases(args.manual)

    print(f"🔍  Discovering terms for {brand} …")

    words_top   = top_token_scan(brand)
    words_iter  = iterative_extension(brand, words_top)
    words_highT = high_t_sampling(brand)
    rivals      = competitor_sampling(brand)

    now = utcnow_iso()
    rows = []

    for phr in sorted(words_top | words_iter | words_highT):
        rows.append((brand, phr, "auto_word",       None, now))
    for riv in sorted(rivals):
        rows.append((brand, riv, "auto_competitor", None, now))
    for phr in sorted(manual_set):
        rows.append((brand, phr, "manual",          None, now))

    save_csv(rows, args.outfile)
    print(f"✅  {len(rows):,} phrases written → {args.outfile}")

if __name__ == "__main__":
    main()