"""Final model production: train the winning method on the full dataset,
export the model (weights + config), and serve a real playground.

No fake predictions — the playground uses the actual trained weights.
"""

import json
import os
import time

import numpy as np

from .data import load_dataset
from .genome import Genome
from .models import build_model, softmax
from .metrics import classification_report, log_loss, confusion_matrix
from .trainer import Trainer


def train_final_model(config, genome_dict, out_dir="models/best_model"):
    """Train the winning genome on the full dataset with a generous budget.

    Returns metadata dict. The model is saved as .npz (portable) plus
    metadata.json / training_config.json / research_report.md.
    """
    os.makedirs(out_dir, exist_ok=True)
    ds = load_dataset(config["dataset"], max_rows=config.get("max_rows"))
    genome = Genome.from_dict(genome_dict)
    seed = config.get("seed", 0)

    model = build_model(config.get("architecture", "mlp"),
                        ds.X.shape[1], ds.n_classes, seed=seed)
    budget = {"epochs": config.get("final_epochs", 30),
              "max_seconds": config.get("final_max_seconds", 1800)}
    trainer = Trainer(genome, model, ds, budget, seed=seed)
    metrics = trainer.train()

    # save weights (.npz = numpy portable format)
    weights_path = os.path.join(out_dir, "model.npz")
    np.savez(weights_path, **model.state_dict())

    meta = {
        "name": genome.name,
        "genome": genome.genes,
        "hypothesis": genome.hypothesis,
        "dataset": config["dataset"],
        "architecture": config.get("architecture", "mlp"),
        "n_classes": ds.n_classes,
        "label_names": ds.label_names,
        "in_dim": int(ds.X.shape[1]),
        "metrics": {k: v for k, v in metrics.items() if k != "history"},
        "trained_at": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime()),
        "param_count": model.param_count(),
    }
    with open(os.path.join(out_dir, "metadata.json"), "w") as f:
        json.dump(meta, f, indent=2, default=str)
    with open(os.path.join(out_dir, "training_config.json"), "w") as f:
        json.dump({"config": config, "budget": budget}, f, indent=2, default=str)

    # export a self-contained predictor module for the playground
    with open(os.path.join(out_dir, "predictor.py"), "w") as f:
        f.write(PREDICTOR_TEMPLATE)

    return meta, metrics


PREDICTOR_TEMPLATE = '''"""Self-contained playground predictor (auto-generated).

Loads model.npz + metadata.json and runs real forward passes.
Usage:
    from predictor import Predictor
    p = Predictor(".")
    p.predict_text("some text")   # for text datasets
    p.predict_vector([...])       # for numeric datasets
"""
import json
import os

import numpy as np

import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
from engine.models import build_model, softmax
from engine.data import text_to_vector, l2_normalize_rows


class Predictor:
    def __init__(self, model_dir):
        with open(os.path.join(model_dir, "metadata.json")) as f:
            self.meta = json.load(f)
        self.model = build_model(self.meta["architecture"], self.meta["in_dim"],
                                 self.meta["n_classes"], seed=0)
        z = np.load(os.path.join(model_dir, "model.npz"))
        self.model.load_state_dict({k: z[k] for k in z.files})

    def predict_vector(self, vec):
        x = np.asarray(vec, dtype=np.float32).reshape(1, -1)
        logits = self.model.forward(x)
        probs = softmax(logits)[0]
        idx = int(np.argmax(probs))
        return {"label": self.meta["label_names"][idx],
                "label_index": idx,
                "confidence": float(probs[idx]),
                "probabilities": {n: float(p)
                                  for n, p in zip(self.meta["label_names"], probs)}}

    def predict_text(self, text):
        v = text_to_vector(text, self.meta["in_dim"])
        v = l2_normalize_rows(v.reshape(1, -1))[0]
        return self.predict_vector(v)
'''


def load_predictor(model_dir="models/best_model"):
    """Load the trained model for the playground (in-process)."""
    with open(os.path.join(model_dir, "metadata.json")) as f:
        meta = json.load(f)
    model = build_model(meta["architecture"], meta["in_dim"],
                        meta["n_classes"], seed=0)
    z = np.load(os.path.join(model_dir, "model.npz"))
    model.load_state_dict({k: z[k] for k in z.files})
    return meta, model


def predict(meta, model, text=None, vector=None):
    """Real prediction using the actual trained weights."""
    if text is not None:
        from .data import text_to_vector, l2_normalize_rows
        v = text_to_vector(text, meta["in_dim"])
        v = l2_normalize_rows(v.reshape(1, -1))
    elif vector is not None:
        v = np.asarray(vector, dtype=np.float32).reshape(1, -1)
    else:
        raise ValueError("Provide text or vector")
    logits = model.forward(v)
    probs = softmax(logits)[0]
    idx = int(np.argmax(probs))
    return {"label": meta["label_names"][idx], "label_index": idx,
            "confidence": float(probs[idx]),
            "probabilities": {n: float(p) for n, p in
                              zip(meta["label_names"], probs)}}
