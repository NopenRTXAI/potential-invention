"""Conversational data pipeline + training + inference for TinyGPT.

Data: DailyDialog (real human two-party conversations, via pyarrow or the
bundled parquet_lite fallback).

Format per dialogue:
    <u> user turn <eot> <a> assistant turn <eot> ... <eos>
The model is trained on full dialogues. At inference we feed the history
and generate the assistant's next turn after an <a> marker.

This is a real language model trained by next-token prediction — replies
are genuinely generated, not retrieved.
"""

import os
import re
import json
import time
import numpy as np

from .gpt import TinyGPT, AdamW, softmax

DATA = os.path.join(os.path.dirname(__file__), "..", "data",
                    "dailydialog.parquet")
CURATED = os.path.join(os.path.dirname(__file__), "..", "data", "curated",
                       "curriculum.jsonl")
CURATED_SEED = os.path.join(os.path.dirname(__file__), "..", "data", "curated",
                            "seed_dialogues.jsonl")
MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models",
                         "chat_model")

# special tokens
PAD, U_TOK, A_TOK, EOT, EOS, UNK = 0, 1, 2, 3, 4, 5
SPECIALS = ["<pad>", "<u>", "<a>", "<eot>", "<eos>", "<unk>"]
# Unicode words preserve Arabic, accented French, English, and code symbols.
WORD_RE = re.compile(
    r"[^\W_]+(?:['’][^\W_]+)?|[.,!?;:(){}\[\]<>+=*/%#@&|^~`-]",
    re.UNICODE)


# ---------------------------------------------------------------- tokenizer

class Tokenizer:
    def __init__(self, vocab=None):
        self.vocab = vocab or {}   # word -> id (ids start at 5)
        self.inv = {}

    def build(self, texts, min_count=2, max_vocab=9000):
        counts = {}
        for t in texts:
            for w in WORD_RE.findall(t.lower()):
                counts[w] = counts.get(w, 0) + 1
        words = sorted((w for w, c in counts.items() if c >= min_count),
                       key=lambda w: -counts[w])[:max_vocab]
        self.vocab = {w: i + len(SPECIALS) for i, w in enumerate(words)}
        self.inv = {i: w for w, i in self.vocab.items()}
        return self

    def encode(self, text):
        text = text.lower().replace("’", "'")
        return [self.vocab.get(w, UNK) for w in WORD_RE.findall(text)]

    def decode(self, ids):
        out = []
        for i in ids:
            if i < len(SPECIALS):
                if i == EOT or i == EOS:
                    out.append("\n")
                continue
            out.append(self.inv.get(i, "?"))
        text = " ".join(out)
        text = re.sub(r" ([.,!?;:])", r"\1", text)
        text = re.sub(r" ([)}\]])", r"\1", text)
        text = re.sub(r"([(\[{]) ", r"\1", text)
        text = re.sub(r" (['’])", r"\1", text)
        return text.strip()

    @property
    def size(self):
        return len(self.vocab) + len(SPECIALS)


# ---------------------------------------------------------------- data

def load_dialogues(path=DATA, limit=None, include_extra=True,
                   include_curated=True):
    """Return list of dialogues; each is list of utterance strings.

    Loads DailyDialog (always) + SODA (data/extra/soda_0.parquet, if present).
    """
    rows = None
    try:
        import pyarrow.parquet as pq
        t = pq.read_table(path)
        rows = t.to_pylist()
    except Exception:
        from .parquet_lite import read_parquet
        rows = read_parquet(path)
    dialogs = {}
    for r in rows:
        dialogs.setdefault(r["dialog_id"], []).append(r["utterance"].strip())
    out = [v for _, v in sorted(dialogs.items())]
    if include_extra:
        out.extend(load_soda(limit=limit))
    if include_curated:
        out.extend(load_curated())
    if limit:
        out = out[:limit]
    return out


def load_curated(path=CURATED):
    """Load reviewed multilingual identity, language, and coding examples."""
    if not os.path.exists(path):
        path = CURATED_SEED
    if not os.path.exists(path):
        return []
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            try:
                row = json.loads(line)
                dialogue = [str(x).strip() for x in row["dialogue"]]
                if len(dialogue) >= 2 and all(dialogue):
                    out.append(dialogue)
            except (ValueError, KeyError, TypeError):
                continue
    return out


def load_soda(path=None, limit=None):
    """Load SODA dialogues (allenai/soda) — 440k real-style dialogues."""
    import glob as _glob
    paths = [path] if path else sorted(
        _glob.glob(os.path.join(os.path.dirname(DATA), "extra",
                                "soda_*.parquet")))
    out = []
    for p in paths:
        try:
            import pyarrow.parquet as pq
            t = pq.read_table(p, columns=["dialogue"])
            for r in t.to_pylist():
                dl = r.get("dialogue") or []
                dl = [u.strip() for u in dl if u and u.strip()]
                if len(dl) >= 2:
                    out.append(dl)
        except Exception as e:
            print(f"soda load skipped ({p}): {e}", flush=True)
    if limit:
        out = out[:limit]
    return out


def dialogue_to_ids(tok, utts):
    """One dialogue -> token list with speaker markers."""
    ids = []
    for i, u in enumerate(utts):
        speaker = U_TOK if i % 2 == 0 else A_TOK
        ids.append(speaker)
        ids.extend(tok.encode(u))
        ids.append(EOT)
    ids.append(EOS)
    return ids


def make_samples(tok, dialogues, block):
    """Slide windows over each dialogue -> training samples of length block+1.

    Each sample: x = tokens[:-1], y = tokens[1:] (next-token prediction).
    """
    X, Y = [], []
    for d in dialogues:
        ids = dialogue_to_ids(tok, d)
        if len(ids) < 4:
            continue
        for start in range(0, max(1, len(ids) - block), block // 2):
            chunk = ids[start:start + block + 1]
            if len(chunk) < 8:
                continue
            chunk = chunk + [PAD] * (block + 1 - len(chunk))
            X.append(chunk[:-1])
            Y.append(chunk[1:])
    return np.array(X, dtype=np.int64), np.array(Y, dtype=np.int64)


# ---------------------------------------------------------------- training

def train(tok, X, Y, d_model=160, n_layers=4, n_heads=4, block=128,
          epochs=2, batch=24, lr=3e-4, seed=0, log_every=50, max_minutes=45,
          callback=None, resume=None, val_frac=0.02, ckpt_every_min=15,
          ckpt_dir=None):
    """Train TinyGPT on next-token prediction. Returns (model, history).

    resume: optional (model, opt_state, hist, start_step) tuple to continue
    training a saved run (from chat.resume_state()).
    ckpt_every_min: save a resumable checkpoint every N minutes so an
    interrupted run (server restart, terminal close) continues where it
    left off instead of starting from zero.
    """
    rng = np.random.default_rng(seed)
    if resume is not None:
        model, opt_state, hist, start_step = resume
        opt = AdamW(model.p, lr=lr)
        opt.m, opt.v, opt.t = opt_state["m"], opt_state["v"], opt_state["t"]
        step = start_step
        print(f"resuming from step {step}", flush=True)
    else:
        model = TinyGPT(vocab_size=tok.size, d_model=d_model,
                       n_layers=n_layers, n_heads=n_heads, block=block,
                       seed=seed)
        opt = AdamW(model.p, lr=lr)
        hist = []
        step = 0
    n = X.shape[0]
    t0 = time.time()
    last_ckpt = t0
    for ep in range(epochs):
        order = rng.permutation(n)
        for bstart in range(0, n, batch):
            bi = order[bstart:bstart + batch]
            if len(bi) < 4:
                continue
            xb, yb = X[bi], Y[bi]
            if callback:
                callback({"step": step + 1, "epoch": ep + 1, "loss": None,
                          "elapsed": round(time.time() - t0, 1),
                          "stage": "computing"})
            fwd = model.forward(xb)
            loss, grads = model.backward(xb, yb, fwd)
            opt.step(model.p, grads)
            step += 1
            hist.append({"step": step, "epoch": ep + 1, "loss": loss,
                         "elapsed": round(time.time() - t0, 1)})
            if step % log_every == 0 or step == 1:
                msg = (f"epoch {ep+1} step {step} loss {loss:.3f} "
                       f"ppl {np.exp(min(loss, 20)):.1f} "
                       f"{(time.time()-t0)/60:.1f}min")
                print(msg, flush=True)
                if callback:
                    callback(hist[-1])
            now = time.time()
            if (ckpt_every_min and ckpt_dir and
                    (now - last_ckpt) / 60 >= ckpt_every_min):
                last_ckpt = now
                try:
                    save(model, tok, hist, out_dir=ckpt_dir, opt=opt)
                    print(f"checkpoint saved at step {step}", flush=True)
                except Exception as e:
                    print(f"checkpoint save failed: {e}", flush=True)
            if (time.time() - t0) / 60 > max_minutes:
                print(f"time budget reached at step {step}", flush=True)
                return model, hist
    return model, hist


# ---------------------------------------------------------------- inference

class ChatBot:
    def __init__(self, model, tok, block=128, temperature=0.85, top_k=25):
        self.model = model
        self.tok = tok
        self.block = block
        self.temperature = temperature
        self.top_k = top_k
        self.rng = np.random.default_rng()

    def reply(self, history, max_new=40):
        """history: list of (speaker, text) with speaker in {'u','a'}.
        Returns generated assistant text."""
        ids = []
        for sp, txt in history[-6:]:  # keep recent context
            ids.append(U_TOK if sp == "u" else A_TOK)
            ids.extend(self.tok.encode(txt))
            ids.append(EOT)
        ids.append(A_TOK)  # assistant speaks next
        ids = ids[-self.model.T:]
        out = self.model.generate(np.array([ids], dtype=np.int64),
                                  max_new=max_new,
                                  temperature=self.temperature,
                                  top_k=self.top_k, eos=EOT,
                                  rng=self.rng)
        gen = out[len(ids):]
        # cut at end-of-turn
        if EOT in list(gen):
            gen = list(gen)[:list(gen).index(EOT)]
        return self.tok.decode([int(t) for t in gen])


# ---------------------------------------------------------------- persistence

def save(model, tok, hist, out_dir=MODEL_DIR, opt=None):
    """Save model (fp16) + tokenizer + history + optional optimizer state.

    Optimizer state (Adam moments) is saved fp32 in opt_state.npz so a long
    run can be resumed without losing momentum.
    """
    os.makedirs(out_dir, exist_ok=True)
    np.savez_compressed(os.path.join(out_dir, "model.npz"),
                        **{k: v.astype(np.float16) for k, v in
                           model.p.items()})
    meta = {
        "vocab": tok.vocab,
        "d_model": model.d, "n_layers": model.L, "n_heads": model.H,
        "block": model.T, "vocab_size": model.V,
        "params": model.param_count(),
        "history": hist[-20:],
        "final_loss": hist[-1]["loss"] if hist else None,
    }
    if opt is not None:
        meta["steps"] = int(opt.t)
    with open(os.path.join(out_dir, "metadata.json"), "w") as f:
        json.dump(meta, f)
    if opt is not None:
        np.savez_compressed(
            os.path.join(out_dir, "opt_state.npz"),
            **{f"m.{k}": v for k, v in opt.m.items()},
            **{f"v.{k}": v for k, v in opt.v.items()},
            t=np.int64(opt.t))


def load(out_dir=MODEL_DIR):
    with open(os.path.join(out_dir, "metadata.json")) as f:
        meta = json.load(f)
    tok = Tokenizer(vocab={w: int(i) for w, i in meta["vocab"].items()})
    # rebuild inv
    tok.inv = {i: w for w, i in tok.vocab.items()}
    model = TinyGPT(vocab_size=meta["vocab_size"], d_model=meta["d_model"],
                    n_layers=meta["n_layers"], n_heads=meta["n_heads"],
                    block=meta["block"])
    z = np.load(os.path.join(out_dir, "model.npz"))
    for k in model.p:
        model.p[k] = z[k].astype(np.float32)
    return model, tok, meta


def load_opt_state(out_dir=MODEL_DIR):
    """Load saved AdamW moments for resuming a long run. None if absent."""
    path = os.path.join(out_dir, "opt_state.npz")
    if not os.path.exists(path):
        return None
    z = np.load(path)
    return {"m": {k[2:]: v.astype(np.float32) for k, v in z.items()
                  if k.startswith("m.")},
            "v": {k[2:]: v.astype(np.float32) for k, v in z.items()
                  if k.startswith("v.")},
            "t": int(z["t"])}
