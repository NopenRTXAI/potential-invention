"""Datasets: built-in generators + CSV loading with a shared vocabulary.

Built-ins (all CPU-friendly, deterministic seeds):
  - spam        : SMS spam detection (text -> binary classification)
  - digits      : 8x8 grayscale digits (10 classes)
  - circles     : 2D non-linear toy (binary)
  - spirals      : 2D non-linear toy (3 classes)

CSV loading: expects a text column + label column, builds a hashed
n-gram vocabulary (deterministic, streaming, low-memory).
"""

import csv
import hashlib
import io
import json
import os
import re
import zipfile
import urllib.request

import numpy as np

DATA_DIR = os.environ.get("ENGINE_DATA_DIR", "data")
_TOKEN_RE = re.compile(r"[a-z0-9']+")


# ---------------------------------------------------------------- text utils

def tokenize(text):
    return _TOKEN_RE.findall(str(text).lower())


def _hash_feature(token, n_buckets):
    h = hashlib.md5(token.encode("utf-8")).digest()
    return int.from_bytes(h[:8], "little") % n_buckets


def text_to_vector(text, n_buckets, use_bigrams=True):
    """Deterministic hashed bag-of-ngrams (memory-bounded, CPU-friendly)."""
    vec = np.zeros(n_buckets, dtype=np.float32)
    toks = tokenize(text)
    for t in toks:
        vec[_hash_feature(t, n_buckets)] += 1.0
    if use_bigrams and len(toks) > 1:
        for a, b in zip(toks, toks[1:]):
            vec[_hash_feature(a + "_" + b, n_buckets)] += 1.0
    return vec


def l2_normalize_rows(X):
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return X / norms


# ---------------------------------------------------------------- dataset API

class Dataset:
    def __init__(self, name, X, y, n_classes, task="classification",
                 vectorizer=None, label_names=None, description=""):
        self.name = name
        self.X = X
        self.y = y
        self.n_classes = n_classes
        self.task = task
        self.vectorizer = vectorizer  # callable text -> vector (for playground)
        self.label_names = label_names or [str(i) for i in range(n_classes)]
        self.description = description
        self.version = 1

    def summary(self):
        return {
            "name": self.name,
            "n_samples": int(self.X.shape[0]),
            "n_features": int(self.X.shape[1]),
            "n_classes": self.n_classes,
            "label_names": self.label_names,
            "task": self.task,
            "description": self.description,
            "version": self.version,
        }

    def split(self, train_frac=0.7, val_frac=0.15, seed=0):
        n = len(self.y)
        rng = np.random.RandomState(seed)
        idx = rng.permutation(n)
        n_tr = int(n * train_frac)
        n_va = int(n * val_frac)
        tr = idx[:n_tr]
        va = idx[n_tr:n_tr + n_va]
        te = idx[n_tr + n_va:]
        return ((self.X[tr], self.y[tr]), (self.X[va], self.y[va]),
                (self.X[te], self.y[te]))


def _ensure_data_dir():
    os.makedirs(DATA_DIR, exist_ok=True)
    return DATA_DIR


# ---------------------------------------------------------------- built-ins

def load_spam(n_buckets=1024, max_rows=None):
    """SMS Spam Collection (UCI). Downloads on first use, cached in data/.

    5572 rows, binary: 'spam' / 'ham'. Hashed bigram features.
    """
    path = os.path.join(_ensure_data_dir(), "smsspam.zip")
    if not os.path.exists(path):
        url = "https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip"
        try:
            urllib.request.urlretrieve(url, path)
        except Exception as e:  # offline fallback -> synthetic spam
            return _synthetic_spam(n_buckets, max_rows)
    try:
        texts, labels = [], []
        with zipfile.ZipFile(path) as z:
            name = next(n for n in z.namelist() if n.endswith(".txt"))
            with z.open(name) as f:
                for raw in io.TextIOWrapper(f, encoding="utf-8", errors="ignore"):
                    line = raw.strip()
                    if not line:
                        continue
                    label, text = line.split("\t", 1)
                    labels.append(1 if label == "spam" else 0)
                    texts.append(text)
        if max_rows:
            texts, labels = texts[:max_rows], labels[:max_rows]
        X = np.stack([text_to_vector(t, n_buckets) for t in texts])
        X = l2_normalize_rows(X)
        y = np.array(labels, dtype=np.int64)
        return Dataset("spam", X, y, 2, vectorizer=lambda t: l2_normalize_rows(
            text_to_vector(t, n_buckets)[None, :])[0],
            label_names=["ham", "spam"],
            description="UCI SMS Spam Collection (hashed bigrams)")
    except Exception:
        return _synthetic_spam(n_buckets, max_rows)


def _synthetic_spam(n_buckets=1024, max_rows=2000):
    """Offline fallback: synthetic spam-like text generated from class templates."""
    rng = np.random.RandomState(42)
    spam_words = ["free", "winner", "claim", "prize", "urgent", "cash", "click",
                  "congratulations", "offer", "limited", "bonus", "guaranteed"]
    ham_words = ["meeting", "tomorrow", "project", "thanks", "lunch", "family",
                 "weekend", "schedule", "report", "hello", "regards", "team"]
    texts, labels = [], []
    n = max_rows or 2000
    for i in range(n):
        if rng.rand() < 0.35:
            pool, lab = spam_words, 1
        else:
            pool, lab = ham_words, 0
        k = rng.randint(4, 14)
        texts.append(" ".join(rng.choice(pool, k, replace=True)))
        labels.append(lab)
    X = np.stack([text_to_vector(t, n_buckets) for t in texts])
    X = l2_normalize_rows(X)
    y = np.array(labels, dtype=np.int64)
    return Dataset("spam-synthetic", X, y, 2,
                   vectorizer=lambda t: l2_normalize_rows(
                       text_to_vector(t, n_buckets)[None, :])[0],
                   label_names=["ham", "spam"],
                   description="Synthetic spam-like text (offline fallback)")


def load_digits(n_buckets=0, max_rows=None):
    """sklearn digits: 1797 8x8 images, 10 classes. Embedded copy (no download)."""
    try:
        from sklearn import datasets
        d = datasets.load_digits()
        X = d.data.astype(np.float32) / 16.0
        y = d.target.astype(np.int64)
    except ImportError:
        X, y = _synthetic_digits()
    if max_rows:
        X, y = X[:max_rows], y[:max_rows]
    return Dataset("digits", X, y, 10, label_names=[str(i) for i in range(10)],
                   description="8x8 grayscale digits, 10 classes")


def _synthetic_digits():
    """Tiny embedded fallback for digits (drawn digit templates + noise)."""
    rng = np.random.RandomState(7)
    templates = np.array([
        [0,1,1,1,0, 1,0,0,0,1, 1,0,0,0,1, 1,0,0,0,1, 0,1,1,1,0],  # 0
        [0,0,1,0,0, 0,1,1,0,0, 0,0,1,0,0, 0,0,1,0,0, 0,1,1,1,0],  # 1
        [0,1,1,1,0, 1,0,0,0,1, 0,0,1,1,0, 0,1,0,0,0, 1,1,1,1,1],  # 2
        [1,1,1,1,0, 0,0,0,0,1, 0,0,1,1,0, 0,0,0,0,1, 1,1,1,1,0],  # 3
        [1,0,0,0,1, 1,0,0,0,1, 1,1,1,1,1, 0,0,0,0,1, 0,0,0,0,1],  # 4
        [1,1,1,1,1, 1,0,0,0,0, 1,1,1,1,0, 0,0,0,0,1, 1,1,1,1,0],  # 5
        [0,1,1,1,0, 1,0,0,0,0, 1,1,1,1,0, 1,0,0,0,1, 0,1,1,1,0],  # 6
        [1,1,1,1,1, 0,0,0,0,1, 0,0,0,1,0, 0,0,1,0,0, 0,1,0,0,0],  # 7
        [0,1,1,1,0, 1,0,0,0,1, 0,1,1,1,0, 1,0,0,0,1, 0,1,1,1,0],  # 8
        [0,1,1,1,0, 1,0,0,0,1, 0,1,1,1,1, 0,0,0,0,1, 0,1,1,1,0],  # 9
    ], dtype=np.float32)
    X, y = [], []
    for cls in range(10):
        for _ in range(60):
            img = templates[cls] + rng.normal(0, 0.25, 25)
            X.append(img)
            y.append(cls)
    return np.array(X), np.array(y, dtype=np.int64)


def load_circles(n_buckets=0, max_rows=1200):
    t = np.linspace(0, 2 * np.pi, max_rows // 2)
    r1, r2 = 1.0, 3.0
    c1 = np.stack([r1 * np.cos(t), r1 * np.sin(t)], 1) + np.random.RandomState(0).normal(0, 0.25, (len(t), 2))
    c2 = np.stack([r2 * np.cos(t), r2 * np.sin(t)], 1) + np.random.RandomState(1).normal(0, 0.25, (len(t), 2))
    X = np.concatenate([c1, c2]).astype(np.float32)
    y = np.concatenate([np.zeros(len(t)), np.ones(len(t))]).astype(np.int64)
    return Dataset("circles", X, y, 2, label_names=["inner", "outer"],
                   description="Concentric circles, non-linear binary")


def load_spirals(n_buckets=0, max_rows=1500):
    rng = np.random.RandomState(3)
    n = max_rows // 3
    X, y = [], []
    for cls in range(3):
        t = np.linspace(0.3, 3.5 * np.pi, n) + rng.rand(n) * 0.4
        r = t / (3.5 * np.pi) * 4 + 0.5
        X.append(np.stack([r * np.cos(t + cls * 2.09), r * np.sin(t + cls * 2.09)], 1))
        y.append(np.full(n, cls))
    X = np.concatenate(X).astype(np.float32) + rng.normal(0, 0.08, (3 * n, 2)).astype(np.float32)
    y = np.concatenate(y).astype(np.int64)
    return Dataset("spirals", X, y, 3, label_names=["a", "b", "c"],
                   description="3-arm spirals, non-linear multi-class")


BUILTIN_DATASETS = {
    "spam": load_spam,
    "digits": load_digits,
    "circles": load_circles,
    "spirals": load_spirals,
}


# ---------------------------------------------------------------- CSV loading

def load_csv(path, text_column=None, label_column=None, n_buckets=1024,
             max_rows=None):
    """Load a CSV with a text column and a label column.

    Auto-detects columns if not given. Labels may be strings (mapped to ids).
    Returns a Dataset with hashed text features.
    """
    with open(path, "r", encoding="utf-8", errors="ignore", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    if not rows:
        raise ValueError("CSV is empty")
    cols = list(rows[0].keys())

    if label_column is None:
        for cand in ("label", "y", "class", "target", "category"):
            if cand in cols:
                label_column = cand
                break
        else:
            raise ValueError(f"No label column found in {cols}")
    if text_column is None:
        for cand in ("text", "message", "content", "body", "sentence", "review"):
            if cand in cols:
                text_column = cand
                break
        else:
            # longest average string column
            best, best_len = None, -1
            for c in cols:
                if c == label_column:
                    continue
                avg = np.mean([len(str(r.get(c, ""))) for r in rows[:50]])
                if avg > best_len:
                    best, best_len = c, avg
            text_column = best

    labels_raw = [str(r[label_column]).strip() for r in rows]
    uniq = sorted(set(labels_raw))
    label_map = {u: i for i, u in enumerate(uniq)}
    y = np.array([label_map[l] for l in labels_raw], dtype=np.int64)
    if max_rows:
        rows = rows[:max_rows]
        y = y[:max_rows]

    X = np.stack([text_to_vector(str(r[text_column]), n_buckets) for r in rows])
    X = l2_normalize_rows(X)
    return Dataset(
        os.path.splitext(os.path.basename(path))[0], X, y, len(uniq),
        vectorizer=lambda t: l2_normalize_rows(text_to_vector(t, n_buckets)[None, :])[0],
        label_names=uniq,
        description=f"CSV {os.path.basename(path)}: '{text_column}' -> '{label_column}'",
    )


def load_dataset(name, **kwargs):
    if name in BUILTIN_DATASETS:
        return BUILTIN_DATASETS[name](**kwargs)
    if os.path.exists(name):
        return load_csv(name, **kwargs)
    raise ValueError(f"Unknown dataset '{name}'. Built-ins: {list(BUILTIN_DATASETS)} or a CSV path.")
