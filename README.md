# Autonomous CPU Training Research Engine

A self-directed research system that **invents, implements, and scientifically
evaluates genuinely new training algorithms for CPUs** — including weak or
low-end CPUs — and a React dashboard to observe and steer it.

No GPU. No ML frameworks. The entire learning stack is **pure NumPy**.

## What it does

The engine runs an autonomous research loop:

```
OBSERVE → HYPOTHESIS → DESIGN → IMPLEMENT → EXPERIMENT → MEASURE
        → COMPARE → ANALYZE → REPEAT
```

- **Genome interpreter** — every candidate algorithm is a *genome* (~24 genes:
  gradient mode, update rule, selection strategy, LR schedule, state precision,
  freeze/curriculum policies…). The `Trainer` executes any genome directly, so
  every invented algorithm is real, runnable code — never a description.
- **Evolutionary search** — targeted hypothesis genomes seed the population;
  mutation, crossover, and random exploration generate new algorithms with
  fully recorded lineage (parents, exact gene changes, scores).
- **Honest science** — identical budgets, splits, and seeds for every algorithm;
  a `ResearchCritic` enforces statistical significance (binomial CI), demands
  multi-seed validation before any "promising" verdict, and checks
  compute-fairness. **No fake novelty**: if nothing beats the baselines, the
  report says so.
- **Failure analysis** — every failed experiment is kept forever in the research
  database and automatically diagnosed (divergence, no learning, slow
  convergence, overfitting…) with suggested next steps.
- **Ablations** — novel mechanisms are neutralized one at a time to measure what
  actually contributes.
- **Real model output** — the winning algorithm trains a final model saved as
  `model.npz` + `metadata.json` + a standalone `predictor.py`.
- **Weak-hardware mode** — on ≤2 cores or <5 GB RAM the engine prioritizes
  memory-saving mechanisms (int8/fp16 optimizer state with sqrt-compression,
  sparse gradients, partial updates).

## Novel mechanism space (all implemented)

| Category | Mechanisms |
|---|---|
| Gradient modes | full, top-k sparse, sign, stale-reuse, momentum-predicted, random-projection, block-sparse |
| Update rules | SGD, momentum, adaptive (Adam-style), sign, normalized, trust-ratio, hybrid adaptive-momentum |
| Update selection | all, importance top-k, layer-cyclic, block-cyclic, loss-gated |
| LR schedules | constant, cosine, warmup-cosine, one-cycle, adaptive-plateau, loss-signal, warm-restarts |
| Memory | fp32 / fp16 / int8 optimizer state (sqrt-compressed second moment), periodic state reset |
| Structured | freeze/unfreeze schedules, curriculum ordering, gradient accumulation, micro-batching |

## Quick start

```bash
# 1. Research engine (pure NumPy + FastAPI)
pip install fastapi uvicorn
python -m engine.server --port 8000

# 2. Dashboard (React + Vite)
npm install
npm run dev          # http://localhost:3000, proxies /api → :8000
```

Then open the dashboard → **Research** → pick a dataset and budget →
**Start Autonomous Research**. Watch algorithms evolve in **Algorithms**,
experiments accumulate in **Experiments**, and when the session ends, test the
final model with real predictions in **Playground** and read the honest
**Report**.

### Headless mode

```bash
python -m engine.server --research --dataset digits --seconds 3600
```

Runs a full autonomous session, trains the final model, and writes
`models/best_model/research_report.md`.

## Datasets

Built-in: `spam` (UCI SMS, text), `digits` (10-class), `circles` (non-linear
binary), `spirals` (deliberately hard 3-class). All use fixed 70/15/15 splits
with seeded shuffling; offline synthetic fallbacks are embedded so the system
works without network access.

## Research database

```
research/
  experiments/   exp_<id>.json   — every experiment, including failures
  lineage/        lineage.json    — full algorithm ancestry graph
  journals/       journal.jsonl   — append-only research log
  results/        leaderboard.json
  checkpoints/                    — automatic resume support
```

## Verified example results

On the `digits` dataset (5-epoch budget, identical for all methods), the
engine discovered `XO-G25-7698` — stale-gradient reuse + hybrid
adaptive-momentum updates + one-cycle LR + quarter-frequency updates —
validated at **0.977 mean test accuracy across 3 seeds vs 0.970 for the best
baseline (SGD-Momentum)**, with ~75% fewer optimizer-state bytes. The
`hybrid_adaptive_momentum` update rule alone reached 0.9815 vs AdamW's 0.9631.

## Dashboard sections

Research · Algorithms (lineage graph) · Experiments · Models · Datasets ·
Hardware · Leaderboard · Research Journal · Playground (real predictions) ·
Report · Settings

## Honest limitations

- Discoveries are dataset- and budget-scoped; multi-seed validation guards
  against noise but small test sets still bound statistical power.
- `int8` optimizer state is lossy (≈0.52 vs 0.96 on digits) — kept as real
  research data, not hidden.
- Some mechanisms (e.g. random-projection gradients) simply fail; they remain
  in the database as negative results.

## TinyGPT — real conversational model (pure NumPy)

A real GPT-style transformer trained from scratch on this machine, no deep
learning framework — every matmul, attention head, and gradient is hand-written
NumPy, numerically verified (relative error ~1e-8 vs finite differences).

- **Architecture:** 3 layers, 4 heads, d_model=128, block=96, weight-tied
  output head → **1,248,128 parameters**
- **Data:** 11,118 real human dialogues (DailyDialog via HuggingFace
  `pixelsandpointers/better_daily_dialog`) → 18,305 training samples,
  5,006-word vocabulary
- **Training:** AdamW, 4 epochs / 2,288 steps / ~38 min on 4 CPU cores,
  cross-entropy loss **9.1 → 3.46**
- **Inference:** speaker-aware autoregressive sampling (temperature 0.85,
  top-k 25) with conversation history

### Try it

Dashboard → **Chat** section, or:

```bash
curl -X POST localhost:8000/api/chat -H 'Content-Type: application/json' \
     -d '{"message": "hello how are you", "history": []}'
```

### Rebuild from scratch

```bash
# data/ and models/ are gitignored — re-fetch the dataset:
curl -L -o data/dailydialog.parquet \
  'https://huggingface.co/api/datasets/pixelsandpointers/better_daily_dialog/parquet/default/train/0.parquet'

# then retrain (background thread, watch progress at /api/chat/status):
curl -X POST localhost:8000/api/chat/train -H 'Content-Type: application/json' \
     -d '{"minutes": 45, "epochs": 4}'
```

### Honest quality expectations

This is a 1.25M-parameter model trained 38 minutes on 4 CPU cores. It produces
real generated English (not templates), but replies are often grammatically
broken or off-topic — ChatGPT-class quality needs billions of parameters and
GPU-scale training. Longer retraining (e.g. `minutes: 120`) improves fluency
but will not change that order of magnitude.

### Hugging Face streaming datasets

For very large corpora, use `engine.hf_stream.stream_dialogues()` with a
specific Hugging Face dataset ID. It uses `streaming=True`, yields small
batches, and does not materialize the corpus on disk:

```bash
pip install -r requirements.txt
python -c "from engine.hf_stream import stream_stats; print(stream_stats('DATASET_ID', max_dialogues=1000))"
```

Streaming avoids a full 12GB download, but the examples still pass through
the network and CPU. Check the dataset license and schema before training.
The adapter supports common `text`, `dialogue`, `conversation`, and
`messages` fields; provide the exact dataset ID before a long run.
