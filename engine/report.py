"""Final research report generator (research_report.md)."""

import json
import os
import time


def generate_report(db, agent_summary, out_path):
    """Write a serious research report from the research database."""
    experiments = db.load_experiments()
    lineage = db.load_lineage()["algorithms"]
    leaderboard = db.load_leaderboard()

    best = agent_summary.get("best") or {}
    best_bl = agent_summary.get("best_baseline") or {}
    cfg = agent_summary.get("config", {})

    # ---- leaderboard rows (unique best per algorithm)
    rows = {}
    for e in experiments:
        if e.get("score") is None:
            continue
        name = e["name"]
        if name not in rows or e["score"] > rows[name]["score"]:
            rows[name] = {"name": name, "score": e["score"],
                          "kind": e.get("kind"),
                          "ram_mb": e.get("metrics", {}).get("peak_rss_mb"),
                          "seconds": e.get("metrics", {}).get("train_seconds"),
                          "epochs": e.get("metrics", {}).get("epochs_completed")}
    top = sorted(rows.values(), key=lambda r: -r["score"])[:15]

    lines = []
    A = lines.append
    A("# Autonomous CPU Training Research Report")
    A("")
    A(f"*Generated: {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}*")
    A("")
    A("## Research Objective")
    A("")
    A("Discover whether fundamentally different training procedures can make "
      "useful machine-learning models substantially more feasible to train on "
      "CPUs, particularly low-resource CPUs, by trading computation time for "
      "memory efficiency, convergence quality, and final model quality.")
    A("")
    A("## Configuration")
    A("")
    A(f"- **Dataset**: `{cfg.get('dataset')}`")
    A(f"- **Architecture**: `{cfg.get('architecture', 'mlp')}`")
    A(f"- **Research budget**: {cfg.get('research_seconds', 0)/60:.0f} minutes")
    A(f"- **Priority**: {cfg.get('priority', 'quality')}")
    A("")
    A("## Hardware")
    A("")
    hw = agent_summary.get("hardware", {})
    A(f"- Platform: {hw.get('platform', 'n/a')}")
    A(f"- Cores: {hw.get('cores', 'n/a')} | SIMD: {hw.get('simd', 'n/a')} | "
      f"RAM: {hw.get('ram_gb', 'n/a')} GB")
    A(f"- Weak-hardware mode: {'YES' if hw.get('weak_hardware') else 'no'}")
    A("")
    A("## Baselines")
    A("")
    for name, info in (agent_summary.get("baseline_scores") or {}).items():
        s = info.get("score") if isinstance(info, dict) else None
        A(f"- **{name}**: {f'{s*100:.2f}%' if s is not None else 'failed'}")
    A("")
    A("## Best Experimental Candidate")
    A("")
    if best:
        A(f"- **Algorithm**: `{best['name']}`")
        A(f"- **Test accuracy**: {best['score']*100:.2f}% "
          f"(best baseline: {best_bl.get('score', 0)*100 if best_bl.get('score') else 'n/a'})")
        A(f"- **Hypothesis**: {best.get('genome', {}).get('hypothesis', '')}")
        A("")
        A("### Update rule (genes)")
        A("")
        A("```json")
        A(json.dumps(best.get("genome", {}).get("genes", {}), indent=2))
        A("```")
    else:
        A("No candidate beat the baselines within the research budget.")
    A("")
    A("## Leaderboard (top 15)")
    A("")
    A("| Method | Score | RAM (MB) | Time (s) | Epochs |")
    A("|---|---|---|---|---|")
    for r in top:
        A(f"| {r['name']} | {r['score']*100:.2f}% | "
          f"{r['ram_mb']:.0f} | {r['seconds']:.1f} | {r['epochs']} |")
    A("")
    A("## Algorithm Lineage")
    A("")
    for name, info in lineage.items():
        parent = info.get("parent")
        mut = info.get("mutation")
        s = info.get("score")
        star = " ⭐" if best and name == best.get("name") else ""
        A(f"- `{name}`{star} — parent: `{parent or '—'}`"
          f"{' — ' + mut if mut else ''}"
          f"{f' — score: {s*100:.2f}%' if s is not None else ''}")
    A("")
    A("## Ablation Studies")
    A("")
    for e in experiments:
        if e.get("kind") == "ablation" and e.get("score") is not None:
            A(f"- `{e['name']}`: {e['score']*100:.2f}%")
    A("")
    A("## Failure Analysis (selected)")
    A("")
    fails = [e for e in experiments if e.get("result") in ("diverged", "error")]
    for e in fails[:10]:
        err = e.get("error") or "diverged"
        A(f"- `{e['name']}` ({e.get('result')}): {err}")
    A("")
    A("## Statistics")
    A("")
    A(f"- Experiments run: {len(experiments)}")
    A(f"- Generations: {agent_summary.get('generation', 0)}")
    A(f"- Promising candidates: {agent_summary.get('promising', 0)}")
    A(f"- Rejected: {agent_summary.get('rejected', 0)}")
    A("")
    A("## Limitations & No Fake Novelty")
    A("")
    A("All candidates are **experimental**. Improvements are reported only "
      "as measured on the held-out test split of this dataset, with the "
      "stated budget and seeds. Statistical significance requires multi-seed "
      "validation (see validation runs in the journal). No claim of global "
      "novelty is made — mechanisms may resemble published techniques.")
    A("")
    A("## Reproducibility")
    A("")
    A("Every experiment (genes, seeds, metrics, lineage) is stored under "
      "`research/experiments/`. The research journal is at "
      "`research/journals/journal.jsonl`. Re-run with the same config and "
      "seed to reproduce.")
    A("")

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        f.write("\n".join(lines))
    return "\n".join(lines)
