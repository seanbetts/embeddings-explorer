# Embeddings Explorer (PoC)

Measure how strongly **GPT‑4o mini** links a brand to the words, phrases and competitors that people normally mention – without needing any internal model embeddings.

---

## 1 Why this exists  🤔

Large language models don’t expose their hidden vector space, but they **do** let us ask for per‑token log‑probabilities (`logprobs`).  
If a phrase is *much* more probable after the prompt “A word I associate with **McDonald’s** is →” than after the neutral prompt “A word I associate with ____ is →”, the model is signalling that it sees a tight association.  
OpenAI’s GPT‑4‑family (incl. **GPT‑4o mini**) surfaces those probabilities via a simple boolean flag.  [oai_citation:0‡OpenAI Cookbook](https://cookbook.openai.com/examples/using_logprobs?utm_source=chatgpt.com)

This PoC turns that signal into a repeatable number (`Δ log P`) and a ranked list you can track over time.

---

## 2 How it works  ⚙️

| Stage | Script | What happens |
|-------|--------|--------------|
|**Discovery**|`discover.py`|• **Top‑token scan** – grab the 20 most‑likely next tokens and keep ones that are brand‑skewed.<br>• **Iterative extension** – grow each token into a short phrase while the skew stays ≥ 0.5.<br>• **High‑T list sampling** – ask GPT‑4o mini for five associated words/phrases 100× at T = 0.8.<br>• **Competitor sampling** – same trick but “companies that compete with…”.<br>• **Manual injection** – any lines in `manual_phrases.txt` are forced into the pool.|
|**Scoring**|`score.py` (coming next)|For each candidate phrase: 1) calculate log P with the brand prompt, 2) log P with the neutral prompt, 3) store the delta (`Δ log P`).|
|**Dashboard**|`dash.py` (optional)|Streamlit front‑end that shows a league‑table and month‑over‑month spark‑lines.|

---

## 3 Project tree  🌳
├── discover.py
├── score.py
├── dash.py
├── data/
│   ├── phrases.csv      # discovered terms (this repo auto‑generates)
│   └── scores.duckdb    # scored results (auto‑generates)
├── .env                 # OPENAI_API_KEY=sk‑…
└── README.md

---

## 4 Quick start  🚀

```bash
git clone https://github.com/your‑org/brand‑probe.git
cd brand‑probe
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt            # openai, python‑dotenv, tiktoken, etc.

echo 'OPENAI_API_KEY=sk‑…' > .env          # or export in your shell

python discover.py --brand "McDonald's"    # fills data/phrases.csv
python score.py    --brand "McDonald's"    # computes Δ log P
streamlit run dash.py                      # visualise
```

## 5 Interpreting Δ log P  📏
	•	**Δ > 1 ** ≈ e²·⁷ times more likely with the brand prompt – strong association
	•	0 < Δ ≤ 1 – moderate link
	•	Δ ≤ 0 – no meaningful link (or even a negative one)

Because both prompts share the same surface structure, subtracting them cancels generic word frequency and isolates the brand effect.

---

## 6 Limitations & gotchas  ⚠️
	•	Tokenisation quirks – smart quotes, apostrophes and ampersands change token IDs; the scripts normalise but double‑check edge cases.
	•	API drift – OpenAI occasionally tweaks field names; the code uses the current logprobs.content[..].top_logprobs schema (May 2025).
	•	Logprob variance – scores fluctuate a bit between runs; we recommend averaging three passes for dashboards.
	•	Model‑specific view – GPT‑4o mini ≠ Claude ≠ Gemini. Run the same pipeline on other providers to spot blind spots.

---

## 7 Road‑map  🗺️
	1.	score.py – batch scorer with DuckDB storage.
	2.	Cross‑model panel – side‑by‑side GPT‑4o mini vs. Mistral Large.
	3.	Sentiment split – add a second prompt (“People feel that …”) to track opinion separately from knowledge.