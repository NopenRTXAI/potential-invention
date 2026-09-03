"""ResearchScientistAgent + ResearchCritic + evolution + ablations + failure analysis.

The autonomous research loop:

  OBSERVE -> FORM HYPOTHESIS -> DESIGN ALGORITHM -> IMPLEMENT -> RUN CONTROLLED
  EXPERIMENT -> MEASURE -> COMPARE WITH BASELINES -> ANALYZE FAILURE ->
  MODIFY HYPOTHESIS -> REPEAT

The agent maintains a research journal, evolves a population of training
algorithms (select / mutate / crossover), runs ablation studies on winners,
applies the self-critic to promising results, and performs automatic failure
analysis. It checkpoints so long research runs resume after crashes.
"""

import json
import os
import random
import time

import numpy as np

from .data import load_dataset
from .db import ResearchDB
from .genome import (Genome, BASELINES, random_genome, genome_diff,
                     describe_genome)
from .models import build_model
from .trainer import Trainer


def _now_str():
    return time.strftime("%Y-%m-%d %H:%M:%S")


# ================================================================ critic

class ResearchCritic:
    """Self-critic: promising results must survive criticism."""

    @staticmethod
    def critique(candidate_metrics, baseline_metrics, n_seeds=1,
                 dataset_n=None):
        """Return list of criticisms (empty = survived)."""
        issues = []
        c = candidate_metrics
        b = baseline_metrics
        if c is None:
            return ["No metrics produced (training failed or diverged)."]
        if c.get("diverged"):
            return ["Training diverged — result invalid."]

        gain = c["test_accuracy"] - b["test_accuracy"]
        if gain <= 0:
            issues.append(
                f"No improvement over baseline ({c['test_accuracy']:.4f} vs "
                f"{b['test_accuracy']:.4f}).")
        # statistical meaningfulness: with a single seed and small test sets,
        # small gains are likely noise. Rough binomial 95% CI half-width.
        if dataset_n:
            n = max(1, int(dataset_n * 0.15))  # test fraction
            se = np.sqrt(max(1e-12, c["test_accuracy"] * (1 - c["test_accuracy"]) / n))
            ci = 1.96 * se
            if gain < ci:
                issues.append(
                    f"Gain {gain:.4f} is within statistical noise "
                    f"(95% CI half-width ±{ci:.4f} on n={n} test samples).")
        if n_seeds < 3 and gain > 0:
            issues.append(
                "Single-seed comparison — repeat with multiple seeds before "
                "claiming superiority.")
        if c.get("train_seconds", 0) > 10 * max(0.1, b.get("train_seconds", 0.1)):
            issues.append(
                "Candidate used far more computation than baseline — "
                "verify the gain is not merely 'more compute'.")
        if c.get("epochs_completed", 0) < 3:
            issues.append("Very short run — result may be undertrained.")
        return issues

    @staticmethod
    def verdict(issues):
        """'promising' | 'needs-validation' | 'rejected'."""
        if not issues:
            return "promising"
        hard = [i for i in issues if "diverged" in i or "No improvement" in i
                or "failed" in i]
        if hard:
            return "rejected"
        return "needs-validation"


# ================================================================ failure analysis

class FailureAnalyzer:
    """Automatic failure analysis: why did an algorithm fail?"""

    @staticmethod
    def analyze(metrics, genome):
        if metrics is None:
            return {"cause": "crash",
                    "explanation": "Training raised an exception or produced no metrics.",
                    "next_step": "Simplify gradient mechanism; add stability guards."}
        reasons = []
        causes = []
        h = metrics.get("history", {})
        tl = h.get("train_loss", [])
        vl = h.get("val_loss", [])

        if metrics.get("diverged"):
            causes.append("divergence")
            if tl and np.isfinite(tl[0]) and tl[-1] > 10 * max(tl[0], 1e-9):
                reasons.append("Loss exploded — update magnitude likely too large.")
            else:
                reasons.append("Loss became non-finite — numerical instability.")
        if tl and len(tl) >= 2:
            if tl[-1] >= tl[0]:
                causes.append("no_learning")
                reasons.append("Training loss did not decrease — updates too small "
                               "or masked out.")
            elif tl[-1] > 0.9 * tl[0]:
                causes.append("slow_convergence")
                reasons.append("Very slow convergence — effective learning rate "
                               "may be too low.")
        if vl and len(vl) >= 3 and min(vl) < vl[-1] * 0.9:
            causes.append("overfitting")
            reasons.append("Validation loss rose while training fell — overfitting.")
        if metrics.get("updated_frac_mean", metrics.get("history", {}).get(
                "updated_frac", [1.0]))[-1:] == [0.0]:
            causes.append("no_updates")
            reasons.append("Parameters were never updated (selection mask empty).")
        if not causes:
            causes.append("underperforming")
            reasons.append("Trained stably but scored below the baseline — "
                           "mechanism combination did not help on this dataset.")

        # Suggest the next modification experimentally
        g = genome.genes
        next_step = None
        if "divergence" in causes:
            next_step = ("Add update normalization (rms_clip) or lower lr; "
                         "avoid aggressive gradient amplification.")
            if g.get("update_norm_mode") == "none":
                next_step += " Candidate lacks update_norm_mode — mutate towards rms_clip."
        elif "no_learning" in causes or "no_updates" in causes:
            next_step = ("Increase update coverage (update_selection=all, "
                         "update_freq=1.0) or raise lr.")
        elif "slow_convergence" in causes:
            next_step = "Raise lr or switch to adaptive update rule."
        elif "overfitting" in causes:
            next_step = "Add weight decay or early stopping via shorter budget."
        else:
            next_step = "Mutate a different mechanism (gradient mode or selection)."
        return {"causes": causes, "explanation": " ".join(reasons) or "Unknown.",
                "next_step": next_step}


# ================================================================ agent

class ResearchScientistAgent:
    """The autonomous research loop driver."""

    def __init__(self, config, db=None):
        self.cfg = config
        self.db = db or ResearchDB(config.get("research_dir", "research"))
        self.rng = random.Random(config.get("seed", 0))
        self.hw = config.get("hardware") or {}
        self.population = []          # list of Genome
        self.archive = {}             # name -> {"genome": dict, "score": float, ...}
        self.baseline_scores = {}
        self.generation = 0
        self.experiment_count = 0
        self.promising = []
        self.rejected = 0
        self.best = None
        self.best_baseline = None
        self.pending_validation = []   # candidates awaiting multi-seed validation
        self.validated = {}           # name -> validation result
        self._last_validated = None
        self.start_time = None
        self.stop_requested = False
        self.status = "idle"
        self.run_id = config.get("run_id", f"run_{int(time.time())}")
        self._load_checkpoint()

    # ------------------------------------------------------------ lifecycle

    def request_stop(self):
        self.stop_requested = True

    def _time_left(self):
        if not self.start_time:
            return float("inf")
        elapsed = time.time() - self.start_time
        return self.cfg["research_seconds"] - elapsed

    def _save_checkpoint(self):
        self.db.save_checkpoint(self.run_id, {
            "config": self.cfg,
            "generation": self.generation,
            "experiment_count": self.experiment_count,
            "rejected": self.rejected,
            "population": [g.to_dict() for g in self.population],
            "archive": {k: v for k, v in self.archive.items()},
            "baseline_scores": self.baseline_scores,
            "best": self.best,
            "best_baseline": self.best_baseline,
            "promising": [p["name"] for p in self.promising],
            "elapsed_seconds": (time.time() - self.start_time) if self.start_time else 0,
        })

    def _load_checkpoint(self):
        state = self.db.load_checkpoint(self.run_id)
        if not state:
            return
        try:
            self.generation = state.get("generation", 0)
            self.experiment_count = state.get("experiment_count", 0)
            self.rejected = state.get("rejected", 0)
            self.population = [Genome.from_dict(d) for d in state.get("population", [])]
            self.archive = state.get("archive", {})
            self.baseline_scores = state.get("baseline_scores", {})
            self.best = state.get("best")
            self.best_baseline = state.get("best_baseline")
            self.promising = [{"name": n} for n in state.get("promising", [])]
            self.cfg.update(state.get("config", {}))
            self.db.append_journal({"type": "checkpoint_resumed",
                                    "run_id": self.run_id,
                                    "generation": self.generation})
        except Exception as e:
            self.db.append_journal({"type": "checkpoint_error", "error": str(e)})

    # ------------------------------------------------------------ experiments

    def _budget_for(self, kind):
        """Training budget scaled to research phase and hardware."""
        weak = self.hw.get("weak_hardware", False)
        base_epochs = self.cfg.get("epochs", 8)
        if kind == "baseline":
            epochs = base_epochs
        elif kind == "candidate":
            epochs = base_epochs
        else:  # ablation / validation
            epochs = max(4, base_epochs // 2)
        max_seconds = self.cfg.get("max_train_seconds", 120)
        if weak:
            max_seconds = min(max_seconds, 90)
        return {"epochs": epochs, "max_seconds": max_seconds}

    def _run_one(self, genome, kind="candidate", seed=None):
        """Run one controlled experiment. Returns (metrics, record) or (None, record)."""
        seed = self.cfg.get("seed", 0) if seed is None else seed
        exp_id = self.experiment_count + 1
        self.experiment_count += 1
        t0 = time.time()
        record = {
            "id": exp_id,
            "name": genome.name,
            "kind": kind,
            "generation": genome.generation,
            "hypothesis": genome.hypothesis,
            "genes": genome.genes,
            "parent": genome.parent,
            "mutation": genome.mutation,
            "seed": seed,
            "dataset": self.cfg["dataset"],
            "architecture": self.cfg.get("architecture", "mlp"),
            "started_at": _now_str(),
        }
        metrics = None
        error = None
        try:
            ds = self._dataset()
            model = build_model(self.cfg.get("architecture", "mlp"),
                                ds.X.shape[1], ds.n_classes, seed=seed)
            trainer = Trainer(genome, model, ds, self._budget_for(kind), seed=seed)
            metrics = trainer.train()
            record["metrics"] = {k: v for k, v in metrics.items() if k != "history"}
            record["history"] = metrics.get("history", {})
            record["score"] = metrics["test_accuracy"]
        except Exception as e:
            error = f"{type(e).__name__}: {e}"
            record["error"] = error
        record["wall_seconds"] = time.time() - t0
        record["result"] = ("ok" if metrics and not metrics.get("diverged")
                           else "diverged" if metrics else "error")
        if metrics:
            record["failure_analysis"] = None  # filled by caller if needed
        self.db.save_experiment(record)
        self.db.add_lineage(genome.to_dict(), experiment_id=exp_id,
                            score=record.get("score"))
        self.db.append_journal({
            "type": "experiment",
            "id": exp_id,
            "name": genome.name,
            "kind": kind,
            "result": record["result"],
            "score": record.get("score"),
            "error": error,
        })
        return metrics, record

    def _dataset(self):
        if not hasattr(self, "_ds_cache"):
            self._ds_cache = load_dataset(self.cfg["dataset"],
                                          max_rows=self.cfg.get("max_rows"))
        return self._ds_cache

    # ------------------------------------------------------------ baselines

    def run_baselines(self):
        """Establish fair baselines: SGD, AdamW, + one more (identical budget)."""
        self.db.append_journal({"type": "phase", "phase": "baselines",
                                "note": "Establishing fair baselines"})
        for name, genes in BASELINES.items():
            if name not in self.baseline_scores:
                genome = Genome(dict(genes), name=name, generation=0,
                                hypothesis=f"Baseline {name}")
                metrics, record = self._run_one(genome, kind="baseline")
                if metrics:
                    self.baseline_scores[name] = {
                        "score": metrics["test_accuracy"],
                        "metrics": {k: v for k, v in metrics.items() if k != "history"},
                    }
                    if self.best_baseline is None or metrics["test_accuracy"] > \
                            self.best_baseline["score"]:
                        self.best_baseline = {
                            "name": name, "score": metrics["test_accuracy"],
                            "genome": genome.to_dict(),
                            "metrics": {k: v for k, v in metrics.items() if k != "history"},
                        }
                else:
                    self.baseline_scores[name] = {"score": None, "error": record.get("error")}
        self.db.append_journal({
            "type": "baselines_complete",
            "scores": {k: (v["score"] if isinstance(v, dict) else None)
                       for k, v in self.baseline_scores.items()},
        })
        self._save_checkpoint()

    # ------------------------------------------------------------ evolution

    def _seed_population(self):
        """Initial population: a few random genomes + mechanism-targeted seeds."""
        self.population = []
        # targeted seeds: one per novel research direction
        targets = [
            ("SparseGrad-v1", {"grad_mode": "topk", "grad_topk_frac": 0.2,
                               "update_rule": "adaptive", "lr": 0.003}),
            ("StaleGrad-v1", {"grad_mode": "stale_reuse", "stale_steps": 2,
                              "update_rule": "momentum", "momentum": 0.9,
                              "lr": 0.03}),
            ("PredictGrad-v1", {"grad_mode": "momentum_predict", "stale_steps": 3,
                                "momentum": 0.9, "update_rule": "adaptive",
                                "lr": 0.003}),
            ("SignGrad-v1", {"grad_mode": "sign", "update_rule": "sign",
                             "momentum": 0.9, "lr": 0.003}),
            ("LayerCycle-v1", {"update_selection": "layer_cyclic",
                               "update_rule": "adaptive", "lr": 0.003}),
            ("BlockCycle-v1", {"update_selection": "block_cyclic",
                               "update_rule": "adaptive", "lr": 0.003}),
            ("MemSave-v1", {"state_precision": "int8", "state_reset_every": 500,
                            "update_rule": "adaptive", "lr": 0.003}),
            ("HybridUpd-v1", {"update_rule": "hybrid_adaptive_momentum",
                              "lr": 0.003, "momentum": 0.95, "beta2": 0.99}),
            ("LossSignal-v1", {"lr_schedule": "loss_signal", "update_rule": "adaptive",
                               "lr": 0.003}),
            ("Progressive-v1", {"freeze_schedule": "freeze_first_then_unfreeze",
                                "curriculum": "easy_first", "update_rule": "adaptive",
                                "lr": 0.003}),
        ]
        base = dict(BASELINES["AdamW"])
        for name, over in targets:
            genes = dict(base)
            genes.update(over)
            g = Genome(genes, name=name, generation=1,
                       hypothesis=f"Targeted hypothesis: {describe_genome(genes)}")
            self.population.append(g)
        # plus random explorers
        while len(self.population) < self.cfg.get("population_size", 12):
            self.population.append(random_genome(self.rng, generation=1))
        self.db.append_journal({"type": "phase", "phase": "seed_population",
                                "size": len(self.population)})

    def _evolve_step(self):
        """One generation: evaluate, select, mutate, crossover."""
        self.generation += 1
        self.db.append_journal({"type": "generation", "generation": self.generation,
                                "population": len(self.population)})

        # ---- evaluate population
        scored = []
        for genome in self.population:
            if self.stop_requested or self._time_left() <= 0:
                break
            metrics, record = self._run_one(genome, kind="candidate")
            score = metrics["test_accuracy"] if metrics else None
            analysis = None
            if metrics is None or record["result"] != "ok":
                analysis = FailureAnalyzer.analyze(metrics, genome)
                self.rejected += 1
                self.db.append_journal({
                    "type": "failure_analysis", "name": genome.name,
                    "analysis": analysis})
            scored.append((genome, score, metrics, record, analysis))

        # ---- compare with best baseline
        best_bl = self.best_baseline["score"] if self.best_baseline else 0.0
        survivors = []
        for genome, score, metrics, record, analysis in scored:
            if score is None:
                continue
            self.archive[genome.name] = {
                "genome": genome.to_dict(), "score": score,
                "metrics": {k: v for k, v in metrics.items() if k != "history"},
                "generation": genome.generation,
            }
            issues = ResearchCritic.critique(
                metrics, self.best_baseline["metrics"]
                if self.best_baseline else {"test_accuracy": 0.0, "train_seconds": 1},
                dataset_n=len(self._dataset().y))
            verdict = ResearchCritic.verdict(issues)
            if verdict == "promising":
                self.promising.append({"name": genome.name, "score": score,
                                       "generation": self.generation})
                self.db.append_journal({"type": "promising", "name": genome.name,
                                        "score": score})
                if self.best is None or score > self.best["score"]:
                    self.best = {"name": genome.name, "score": score,
                                 "genome": genome.to_dict(),
                                 "metrics": {k: v for k, v in metrics.items()
                                             if k != "history"}}
            elif verdict == "needs-validation":
                survivors.append((genome, score))
                # queue for multi-seed validation (critic demands it)
                if gain_pending := (score > (self.best_baseline["score"]
                                            if self.best_baseline else 0.0)):
                    self.pending_validation.append(
                        {"name": genome.name, "score": score,
                         "genome": genome.to_dict(),
                         "generation": self.generation})
            else:
                self.rejected += 1

        # ---- select parents (top performers + diversity)
        survivors.sort(key=lambda x: -x[1])
        top = [g for g, s in survivors[:max(2, len(survivors) // 2)]]
        # keep promising archive entries as parents too
        arch_sorted = sorted(self.archive.values(), key=lambda a: -a["score"])
        for a in arch_sorted[:3]:
            g = Genome.from_dict(a["genome"])
            if g.name not in [p.name for p in top]:
                top.append(g)

        # ---- generate next generation: mutate + crossover + explorers
        children = []
        if top:
            for parent in top[: self.cfg.get("population_size", 12) // 2]:
                if self.stop_requested:
                    break
                child = parent.mutate(self.rng)
                child.name = f"{parent.name.split('-')[0]}-G{self.generation}-{self.rng.randint(1000, 9999)}"
                children.append(child)
            if len(top) >= 2:
                for _ in range(max(1, self.cfg.get("population_size", 12) // 4)):
                    if self.stop_requested:
                        break
                    a, b = self.rng.sample(top, min(2, len(top)))
                    child = a.crossover(b, self.rng)
                    child.name = f"XO-G{self.generation}-{self.rng.randint(1000, 9999)}"
                    children.append(child)
        while len(children) < self.cfg.get("population_size", 12):
            children.append(random_genome(self.rng, generation=self.generation))

        self.population = children
        self._save_checkpoint()

    # ------------------------------------------------------------ ablations

    def run_ablation(self, genome_dict):
        """Ablation study: remove components one at a time, measure impact."""
        genome = Genome.from_dict(genome_dict)
        base_metrics, _ = self._run_one(genome, kind="ablation")
        if base_metrics is None:
            return {"error": "base run failed"}
        results = [{"component": "full method",
                    "score": base_metrics["test_accuracy"]}]
        genes = genome.genes
        # ablate towards the AdamW-like neutral for each novel mechanism
        neutral = dict(BASELINES["AdamW"])
        components = []
        if genes.get("grad_mode") != "full":
            components.append(("grad_mode", "full"))
        if genes.get("update_rule") not in ("sgd", "adaptive"):
            components.append(("update_rule", "adaptive"))
        if genes.get("update_selection") != "all":
            components.append(("update_selection", "all"))
        if genes.get("lr_schedule") != "constant":
            components.append(("lr_schedule", "constant"))
        if genes.get("state_precision") != "fp32":
            components.append(("state_precision", "fp32"))
        if genes.get("freeze_schedule") != "none":
            components.append(("freeze_schedule", "none"))
        if genes.get("curriculum") != "none":
            components.append(("curriculum", "none"))
        if genes.get("grad_accum_steps", 1) != 1:
            components.append(("grad_accum_steps", 1))
        for gene, neutral_val in components:
            if self.stop_requested or self._time_left() <= 0:
                break
            g2 = genome.clone()
            g2.genes[gene] = neutral_val
            g2.name = f"{genome.name}-no-{gene}"
            m, _ = self._run_one(g2, kind="ablation")
            results.append({
                "component": f"without {gene}",
                "score": m["test_accuracy"] if m else None,
            })
        self.db.append_journal({"type": "ablation", "name": genome.name,
                                "results": results})
        return {"name": genome.name, "results": results}

    # ------------------------------------------------------------ validation

    def _validate_pending(self):
        """Multi-seed validate queued candidates; promote confirmed winners.

        The critic refuses to call a single-seed result 'promising', so any
        candidate that beats the baseline gets re-run on extra seeds here.
        Promotion requires the mean across seeds to still beat the baseline.
        """
        if not self.best_baseline:
            return
        bl = self.best_baseline["score"]
        # take up to 3 pending, best first
        batch = sorted(self.pending_validation, key=lambda x: -x["score"])[:3]
        self.pending_validation = [p for p in self.pending_validation
                                   if p not in batch]
        for cand in batch:
            if self.stop_requested or self._time_left() <= 0:
                self.pending_validation.append(cand)  # re-queue
                break
            genome = Genome.from_dict(cand["genome"])
            scores = [cand["score"]]
            for s in (1, 2):
                m, _ = self._run_one(genome, kind="validation", seed=s)
                if m:
                    scores.append(m["test_accuracy"])
            mean = float(np.mean(scores))
            out = {"name": genome.name, "seeds": len(scores),
                   "mean": mean, "std": float(np.std(scores)),
                   "scores": scores, "baseline": bl,
                   "confirmed": bool(mean > bl)}
            self.validated[genome.name] = out
            self.db.append_journal({"type": "validation", **out})
            if out["confirmed"] and (self.best is None
                                     or mean > self.best["score"]):
                self.best = {"name": genome.name, "score": mean,
                             "genome": cand["genome"],
                             "metrics": {"test_accuracy": mean,
                                         "validation_std": out["std"],
                                         "seeds": len(scores)}}
                self.promising.append({"name": genome.name, "score": mean,
                                       "generation": self.generation})
                self.db.append_journal({"type": "promising",
                                        "name": genome.name, "score": mean,
                                        "note": "confirmed across seeds"})

    def validate_best(self, n_seeds=3):
        """Repeat-run the best candidate across seeds (critic requirement)."""
        if not self.best:
            return None
        genome = Genome.from_dict(self.best["genome"])
        scores = []
        for s in range(n_seeds):
            if self.stop_requested or self._time_left() <= 0:
                break
            m, _ = self._run_one(genome, kind="validation", seed=s)
            if m:
                scores.append(m["test_accuracy"])
        if not scores:
            return None
        out = {
            "name": genome.name,
            "seeds": len(scores),
            "mean": float(np.mean(scores)),
            "std": float(np.std(scores)),
            "scores": scores,
        }
        self.db.append_journal({"type": "validation", **out})
        return out

    # ------------------------------------------------------------ main loop

    def research(self):
        """Run the autonomous research loop for the full budget."""
        self.start_time = time.time() - (self.cfg.get("_resume_elapsed", 0) or 0)
        self.status = "researching"
        self.db.append_journal({"type": "research_start", "run_id": self.run_id,
                                "config": {k: v for k, v in self.cfg.items()
                                           if k != "hardware"},
                                "hardware": self.hw})
        try:
            if not self.baseline_scores:
                self.run_baselines()
            if not self.population:
                self._seed_population()
            while (self._time_left() > 0 and not self.stop_requested):
                self._evolve_step()
                # periodically: validate pending candidates, ablate the best
                if self.generation % 3 == 0 and self.pending_validation:
                    self._validate_pending()
                if self.best and self.generation % 3 == 0 \
                        and self.best["name"] != self._last_validated:
                    self.run_ablation(self.best["genome"])
                    self.validate_best(n_seeds=2)
                    self._last_validated = self.best["name"]
                self._save_checkpoint()
        except Exception as e:
            self.db.append_journal({"type": "fatal_error", "error": str(e)})
            raise
        finally:
            self.status = "complete" if not self.stop_requested else "stopped"
            self._save_checkpoint()
            self.db.append_journal({"type": "research_end",
                                    "status": self.status,
                                    "experiments": self.experiment_count,
                                    "generations": self.generation})
        return self.summary()

    def summary(self):
        return {
            "status": self.status,
            "run_id": self.run_id,
            "generation": self.generation,
            "experiments": self.experiment_count,
            "promising": len(self.promising),
            "rejected": self.rejected,
            "best_baseline": self.best_baseline,
            "best": self.best,
            "time_elapsed": (time.time() - self.start_time) if self.start_time else 0,
            "time_budget": self.cfg.get("research_seconds", 0),
        }
