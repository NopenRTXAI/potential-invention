"""Training-algorithm genomes — the DNA of a candidate training method.

A genome is an executable specification of a TRAINING PROCEDURE (not just an
optimizer): how gradients are computed, selected, reused, predicted, how
parameters are updated, scheduled, normalized, frozen, and how memory is
managed. The trainer interprets genomes directly, so every generated
candidate is a real, runnable algorithm.

The gene space is deliberately broad so evolution can discover combinations
that no human has written down — while every mechanism is actually
implemented (no fake novelty).
"""

import copy
import hashlib
import random


# ---------------------------------------------------------------- gene space

GENES = {
    # --- gradient computation ---
    "grad_mode": ["full", "topk", "sign", "stale_reuse", "momentum_predict",
                  "rand_proj", "block_sparse"],
    "grad_topk_frac": [0.05, 0.1, 0.2, 0.3, 0.5, 0.8],
    "stale_steps": [1, 2, 3, 4, 8],
    "grad_accum_steps": [1, 2, 4, 8],

    # --- update rule ---
    "update_rule": ["sgd", "momentum", "adaptive", "sign", "normalized",
                    "trust_ratio", "hybrid_adaptive_momentum"],
    "momentum": [0.0, 0.5, 0.9, 0.95, 0.99],
    "beta2": [0.9, 0.95, 0.99, 0.999],
    "eps": [1e-8, 1e-6, 1e-4],
    "weight_decay": [0.0, 1e-5, 1e-4, 1e-3, 1e-2],

    # --- parameter selection / update frequency ---
    "update_selection": ["all", "importance_topk", "layer_cyclic", "block_cyclic",
                         "loss_gated"],
    "update_freq": [0.25, 0.5, 1.0],  # fraction of steps that update params
    "importance_source": ["grad_mag", "grad_history", "adam_diag", "age"],

    # --- learning rate schedule ---
    "lr_schedule": ["constant", "cosine", "warmup_cosine", "one_cycle",
                    "adaptive_plateau", "loss_signal", "warm_restarts"],
    "lr": [0.001, 0.003, 0.01, 0.03, 0.1],
    "lr_min_frac": [0.01, 0.1, 0.3],
    "warmup_frac": [0.0, 0.05, 0.15],

    # --- update normalization / stability ---
    "update_clip": [0.0, 0.5, 1.0, 5.0],  # 0 = off
    "update_norm_mode": ["none", "l2_per_param", "rms_clip", "trust_coef"],

    # --- memory management ---
    "state_precision": ["fp32", "fp16", "int8"],  # optimizer-state precision
    "state_reset_every": [0, 100, 500, 1000],  # 0 = never

    # --- progressive training ---
    "freeze_schedule": ["none", "freeze_first_then_unfreeze",
                        "freeze_last_then_unfreeze", "alternate_freeze"],
    "curriculum": ["none", "easy_first", "loss_sorted"],

    # --- batching / CPU ---
    "micro_batch": [16, 32, 64, 128],
    "shuffle": [True, False],
}

# Genes that must stay coherent (e.g. topk needs a fraction)
DEPENDENCIES = [
    ("grad_mode", "topk", ["grad_topk_frac"]),
    ("grad_mode", "stale_reuse", ["stale_steps"]),
    ("grad_mode", "momentum_predict", ["momentum"]),
    ("update_rule", "adaptive", ["beta2", "eps"]),
    ("update_rule", "hybrid_adaptive_momentum", ["beta2", "eps", "momentum"]),
    ("update_rule", "momentum", ["momentum"]),
    ("update_rule", "sign", ["momentum"]),
    ("update_selection", "importance_topk", ["importance_source"]),
]

# Baselines are fixed genomes (used for fair comparison)
BASELINES = {
    "SGD": {
        "grad_mode": "full", "grad_topk_frac": 0.5, "stale_steps": 1,
        "grad_accum_steps": 1, "update_rule": "sgd", "momentum": 0.0,
        "beta2": 0.999, "eps": 1e-8, "weight_decay": 0.0,
        "update_selection": "all", "update_freq": 1.0,
        "importance_source": "grad_mag", "lr_schedule": "constant",
        "lr": 0.1, "lr_min_frac": 0.1, "warmup_frac": 0.0,
        "update_clip": 0.0, "update_norm_mode": "none",
        "state_precision": "fp32", "state_reset_every": 0,
        "freeze_schedule": "none", "curriculum": "none",
        "micro_batch": 32, "shuffle": True,
    },
    "SGD-Momentum": None,  # filled programmatically below
    "AdamW": None,
    "RMSProp": None,
}


def _baseline(base, **over):
    g = dict(base)
    g.update(over)
    return g


BASELINES["SGD-Momentum"] = _baseline(BASELINES["SGD"], update_rule="momentum",
                                     momentum=0.9, lr=0.03)
BASELINES["AdamW"] = _baseline(BASELINES["SGD"], update_rule="adaptive",
                               lr=0.003, weight_decay=0.01, beta2=0.999,
                               eps=1e-8, lr_schedule="constant")
BASELINES["RMSProp"] = _baseline(BASELINES["SGD"], update_rule="adaptive",
                                 lr=0.003, beta2=0.95, weight_decay=0.0)


# ---------------------------------------------------------------- genome

class Genome:
    """One candidate training algorithm."""

    def __init__(self, genes, name=None, parent=None, parent2=None,
                 mutation=None, generation=0, hypothesis=None, tags=None):
        self.genes = dict(genes)
        self.name = name or self._auto_name()
        self.parent = parent
        self.parent2 = parent2
        self.mutation = mutation  # description of what changed vs parent
        self.generation = generation
        self.hypothesis = hypothesis or self._auto_hypothesis()
        self.tags = tags or []
        self.uid = self._uid()

    # -- identity -----------------------------------------------------------

    def _uid(self):
        h = hashlib.md5(repr(sorted(self.genes.items())).encode()).hexdigest()
        return h[:10]

    def _auto_name(self):
        return "Candidate"

    def _auto_hypothesis(self):
        return describe_genome(self.genes)

    # -- evolution ----------------------------------------------------------

    def mutate(self, rng, n_mutations=None):
        """Return a mutated child genome. Every mutation is recorded."""
        genes = dict(self.genes)
        n = n_mutations or rng.randint(1, 3)
        changes = []
        keys = [k for k in GENES if k in genes]
        for _ in range(n):
            k = rng.choice(keys)
            opts = [o for o in GENES[k] if o != genes[k]]
            if not opts:
                continue
            old = genes[k]
            genes[k] = rng.choice(opts)
            changes.append(f"{k}: {old} -> {genes[k]}")
        # repair coherence
        genes = repair_genome(genes)
        return Genome(genes, parent=self.name, mutation="; ".join(changes),
                      generation=self.generation + 1,
                      hypothesis=f"Mutation of {self.name}: " + "; ".join(changes))

    def crossover(self, other, rng):
        """Uniform crossover: per-gene random parent; keep coherence."""
        genes = {}
        for k in self.genes:
            genes[k] = self.genes[k] if rng.random() < 0.5 else other.genes[k]
        genes = repair_genome(genes)
        return Genome(genes, parent=self.name, parent2=other.name,
                      mutation=f"crossover({self.name}, {other.name})",
                      generation=max(self.generation, other.generation) + 1,
                      hypothesis=f"Combine mechanisms of {self.name} and {other.name}")

    def clone(self):
        return Genome(dict(self.genes), name=self.name, parent=self.parent,
                      mutation=self.mutation, generation=self.generation,
                      hypothesis=self.hypothesis, tags=list(self.tags))

    # -- serialization ------------------------------------------------------

    def to_dict(self):
        return {"name": self.name, "uid": self.uid, "genes": self.genes,
                "parent": self.parent, "parent2": self.parent2,
                "mutation": self.mutation, "generation": self.generation,
                "hypothesis": self.hypothesis, "tags": self.tags}

    @classmethod
    def from_dict(cls, d):
        return cls(d["genes"], name=d["name"], parent=d.get("parent"),
                   parent2=d.get("parent2"), mutation=d.get("mutation"),
                   generation=d.get("generation", 0),
                   hypothesis=d.get("hypothesis"), tags=d.get("tags", []))


def repair_genome(genes):
    """Enforce gene dependencies so genomes stay executable."""
    genes = dict(genes)
    for gene, value, needs in DEPENDENCIES:
        if genes.get(gene) == value:
            for need in needs:
                if need not in genes:
                    genes[need] = random.choice(GENES[need])
    return genes


def random_genome(rng, generation=0):
    genes = {k: rng.choice(v) for k, v in GENES.items()}
    genes = repair_genome(genes)
    # descriptive name from the two most distinctive genes
    tag = f"{genes['grad_mode']}-{genes['update_rule']}"
    name = f"Explorer-{tag}-G{generation}"
    return Genome(genes, name=name, generation=generation,
                  hypothesis="Randomly assembled novel training procedure")


def describe_genome(genes):
    """Human-readable description of what makes this training procedure distinct."""
    parts = []
    gm = genes.get("grad_mode")
    if gm == "full":
        parts.append("exact full gradients")
    elif gm == "topk":
        parts.append(f"sparse top-{int(genes.get('grad_topk_frac', 0.2) * 100)}% gradients")
    elif gm == "sign":
        parts.append("sign-only gradient compression")
    elif gm == "stale_reuse":
        parts.append(f"gradient reuse (recompute every {genes.get('stale_steps', 2)} steps)")
    elif gm == "momentum_predict":
        parts.append("momentum-predicted gradients between exact computations")
    elif gm == "rand_proj":
        parts.append("randomly projected gradient estimation")
    elif gm == "block_sparse":
        parts.append("block-sparse gradient computation")

    ur = genes.get("update_rule")
    if ur == "sgd":
        parts.append("plain additive updates")
    elif ur == "momentum":
        parts.append(f"momentum {genes.get('momentum', 0.9)}")
    elif ur == "adaptive":
        parts.append("diagonally-scaled adaptive updates")
    elif ur == "sign":
        parts.append("sign updates with momentum")
    elif ur == "normalized":
        parts.append("per-parameter normalized updates")
    elif ur == "trust_ratio":
        parts.append("trust-ratio constrained updates")
    elif ur == "hybrid_adaptive_momentum":
        parts.append("hybrid adaptive+momentum update")

    us = genes.get("update_selection")
    if us != "all":
        parts.append(f"{us.replace('_', ' ')} parameter selection")

    if genes.get("update_freq", 1.0) < 1.0:
        parts.append(f"update on {int(genes.get('update_freq') * 100)}% of steps")

    sch = genes.get("lr_schedule")
    if sch != "constant":
        parts.append(f"{sch.replace('_', ' ')} learning-rate schedule")

    if genes.get("state_precision") != "fp32":
        parts.append(f"{genes.get('state_precision')} optimizer state (memory saving)")
    if genes.get("state_reset_every"):
        parts.append(f"state reset every {genes['state_reset_every']} steps")
    if genes.get("freeze_schedule") != "none":
        parts.append(f"{genes['freeze_schedule'].replace('_', ' ')}")
    if genes.get("curriculum") != "none":
        parts.append(f"{genes['curriculum']} curriculum")
    if genes.get("grad_accum_steps", 1) > 1:
        parts.append(f"gradient accumulation x{genes['grad_accum_steps']}")
    return "; ".join(parts) if parts else "vanilla training procedure"


def genome_diff(a, b):
    """What changed between two genomes (for lineage display)."""
    diff = []
    for k in sorted(set(a) | set(b)):
        va, vb = a.get(k, None), b.get(k, None)
        if va != vb:
            diff.append({"gene": k, "from": va, "to": vb})
    return diff
