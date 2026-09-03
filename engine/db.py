"""Research database: permanent storage of every experiment.

Never throws away failed experiments — failures are research data.
Implemented as an append-only JSONL store + JSON index files (no external
database dependency, works on the weakest machines).

Layout:
  research/
    experiments/  exp_<id>.json     one file per experiment (full record)
    lineage/      lineage.json      algorithm ancestry graph
    journals/     journal.jsonl      append-only research journal
    results/      leaderboard.json   current leaderboard
    checkpoints/  <run_id>/         resumable research state
"""

import json
import os
import threading
import time

_LOCK = threading.Lock()


def _now():
    return time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime())


class ResearchDB:
    def __init__(self, root="research"):
        self.root = root
        self.experiments_dir = os.path.join(root, "experiments")
        self.lineage_path = os.path.join(root, "lineage", "lineage.json")
        self.journal_path = os.path.join(root, "journals", "journal.jsonl")
        self.results_dir = os.path.join(root, "results")
        self.checkpoints_dir = os.path.join(root, "checkpoints")
        for d in (self.experiments_dir, os.path.dirname(self.lineage_path),
                  os.path.dirname(self.journal_path), self.results_dir,
                  self.checkpoints_dir):
            os.makedirs(d, exist_ok=True)

    # ------------------------------------------------------------ experiments

    def save_experiment(self, record):
        """record: dict with keys id, name, genome, metrics, ..."""
        record = dict(record)
        record.setdefault("saved_at", _now())
        path = os.path.join(self.experiments_dir, f"exp_{record['id']}.json")
        with _LOCK:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(record, f, indent=2, default=str)

        return path

    def load_experiments(self):
        out = []
        for fn in sorted(os.listdir(self.experiments_dir)):
            if fn.startswith("exp_") and fn.endswith(".json"):
                try:
                    with open(os.path.join(self.experiments_dir, fn), "r",
                              encoding="utf-8") as f:
                        out.append(json.load(f))
                except (OSError, json.JSONDecodeError):
                    continue
        out.sort(key=lambda r: r.get("id", 0))
        return out

    # ------------------------------------------------------------ lineage

    def add_lineage(self, genome_dict, experiment_id=None, score=None):
        """genome_dict: Genome.to_dict()"""
        with _LOCK:
            graph = self.load_lineage()
            graph["algorithms"][genome_dict["name"]] = {
                **genome_dict,
                "experiment_id": experiment_id,
                "score": score,
            }
            os.makedirs(os.path.dirname(self.lineage_path), exist_ok=True)
            with open(self.lineage_path, "w", encoding="utf-8") as f:
                json.dump(graph, f, indent=2, default=str)

    def load_lineage(self):
        if os.path.exists(self.lineage_path):
            try:
                with open(self.lineage_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except (OSError, json.JSONDecodeError):
                pass
        return {"algorithms": {}}

    # ------------------------------------------------------------ journal

    def append_journal(self, entry):
        entry = dict(entry)
        entry.setdefault("ts", _now())
        with _LOCK:
            with open(self.journal_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry, default=str) + "\n")

    def load_journal(self, limit=None):
        entries = []
        if os.path.exists(self.journal_path):
            with open(self.journal_path, "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        entries.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
        return entries[-limit:] if limit else entries

    # ------------------------------------------------------------ leaderboard

    def save_leaderboard(self, rows):
        with _LOCK:
            with open(os.path.join(self.results_dir, "leaderboard.json"),
                      "w", encoding="utf-8") as f:
                json.dump({"updated_at": _now(), "rows": rows}, f, indent=2,
                          default=str)

    def load_leaderboard(self):
        p = os.path.join(self.results_dir, "leaderboard.json")
        if os.path.exists(p):
            try:
                with open(p, "r", encoding="utf-8") as f:
                    return json.load(f)
            except (OSError, json.JSONDecodeError):
                pass
        return {"updated_at": None, "rows": []}

    # ------------------------------------------------------------ checkpoints

    def save_checkpoint(self, run_id, state):
        d = os.path.join(self.checkpoints_dir, run_id)
        os.makedirs(d, exist_ok=True)
        state = dict(state)
        state["saved_at"] = _now()
        with open(os.path.join(d, "state.json"), "w", encoding="utf-8") as f:
            json.dump(state, f, indent=2, default=str)

    def load_checkpoint(self, run_id):
        p = os.path.join(self.checkpoints_dir, run_id, "state.json")
        if os.path.exists(p):
            try:
                with open(p, "r", encoding="utf-8") as f:
                    return json.load(f)
            except (OSError, json.JSONDecodeError):
                return None
        return None

    def clear_checkpoint(self, run_id):
        p = os.path.join(self.checkpoints_dir, run_id, "state.json")
        if os.path.exists(p):
            os.remove(p)
