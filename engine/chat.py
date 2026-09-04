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
MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models",
                         "chat_model")

# special tokens
PAD, U_TOK, A_TOK, EOT, EOS = 0, 1, 2, 3, 4
SPECIALS = ["<pad>", "<u>", "<a>", "<eot>", "<eos>"]
WORD_RE = re.compile(r"[a-z']+|[.,!?;:]")


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
        return [self.vocab.get(w, self.vocab.get("<unk>", 5))
                for w in WORD_RE.findall(text.lower())]

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
        text = re.sub(r"(')", r"\1", text)
        return text.strip()

    @property
    def size(self):
        return len(self.vocab) + len(SPECIALS) + 1  # +1 for <unk>


# ---------------------------------------------------------------- data

def load_dialogues(path=DATA, limit=None):
    """Return list of dialogues; each is list of utterance strings."""
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
          callback=None):
    """Train TinyGPT on next-token prediction. Returns (model, history)."""
    rng = np.random.default_rng(seed)
    model = TinyGPT(vocab_size=tok.size, d_model=d_model, n_layers=n_layers,
                    n_heads=n_heads, block=block, seed=seed)
    opt = AdamW(model.p, lr=lr)
    hist = []
    n = X.shape[0]
    t0 = time.time()
    step = 0
    for ep in range(epochs):
        order = rng.permutation(n)
        for bstart in range(0, n, batch):
            bi = order[bstart:bstart + batch]
            if len(bi) < 4:
                continue
            xb, yb = X[bi], Y[bi]
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

def save(model, tok, hist, out_dir=MODEL_DIR):
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
    with open(os.path.join(out_dir, "metadata.json"), "w") as f:
        json.dump(meta, f)


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
