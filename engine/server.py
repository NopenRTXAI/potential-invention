"""Entry point for the research engine server.

Usage:
    python -m engine.server          # start API server (uvicorn)
    python -m engine.server --research --dataset spam --seconds 3600
                                        # run research headless (no server)
"""

import argparse


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--research", action="store_true",
                    help="run research headless instead of serving")
    ap.add_argument("--dataset", default="spam")
    ap.add_argument("--seconds", type=int, default=3600)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--architecture", default="mlp")
    ap.add_argument("--max-rows", type=int, default=None)
    args = ap.parse_args()

    if args.research:
        from .agent import ResearchScientistAgent
        from .hardware import detect_hardware
        cfg = {
            "dataset": args.dataset,
            "research_seconds": args.seconds,
            "epochs": args.epochs,
            "architecture": args.architecture,
            "max_train_seconds": 120,
            "population_size": 10,
            "priority": "quality",
            "seed": 0,
            "max_rows": args.max_rows,
            "hardware": detect_hardware(),
            "research_dir": "research",
            "run_id": "headless",
        }
        agent = ResearchScientistAgent(cfg)
        summary = agent.research()
        import json
        print(json.dumps(summary, indent=2, default=str))
        # final model: best candidate, or best baseline if none beat it
        winner = agent.best or agent.best_baseline
        if winner:
            from .final_model import train_final_model
            meta, _ = train_final_model(cfg, winner["genome"])
            print("Final model trained:", meta["name"],
                  f"test_acc={meta['metrics']['test_accuracy']:.4f}")
        return

    import uvicorn
    from .server import app
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
"""FastAPI server: research control, dashboard data, playground.

Endpoints:
  GET  /api/status            live research status
  GET  /api/dashboard         full dashboard payload
  GET  /api/experiments       experiment list
  GET  /api/experiments/{id}  one experiment detail
  GET  /api/lineage           algorithm lineage graph
  GET  /api/journal           research journal
  GET  /api/leaderboard       leaderboard
  GET  /api/hardware          hardware info
  GET  /api/datasets          available datasets
  POST /api/research/start    start autonomous research {dataset, seconds, ...}
  POST /api/research/stop     request stop
  GET  /api/model             best model info (if trained)
  POST /api/playground        real prediction with the trained model
  GET  /api/report            final research report (markdown)
"""

import os
import threading
import time
import traceback

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from .agent import ResearchScientistAgent
from .data import BUILTIN_DATASETS
from .db import ResearchDB
from .final_model import train_final_model, load_predictor, predict
from .hardware import detect_hardware
from .report import generate_report

app = FastAPI(title="Autonomous CPU Training Research Engine")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)

STATE = {
    "agent": None,
    "thread": None,
    "hardware": detect_hardware(),
    "final_model": None,   # (meta, model) after training
    "last_error": None,
}


# ---------------------------------------------------------------- models

class ResearchStartReq(BaseModel):
    dataset: str = "spam"
    seconds: int = 3600
    architecture: str = "mlp"
    epochs: int = 8
    max_train_seconds: int = 120
    population_size: int = 10
    priority: str = "quality"
    seed: int = 0
    max_rows: int | None = None


class PlaygroundReq(BaseModel):
    text: str | None = None
    vector: list[float] | None = None


# ---------------------------------------------------------------- helpers

def _agent_summary(agent):
    s = agent.summary()
    s["time_left"] = max(0.0, agent._time_left())
    s["config"] = agent.cfg
    s["hardware"] = agent.hw
    s["baseline_scores"] = agent.baseline_scores
    return s


def _train_final_in_background(agent):
    """After research completes, train the best method into a real model."""
    try:
        winner = agent.best or agent.best_baseline
        if winner:
            meta, metrics = train_final_model(
                agent.cfg, winner["genome"],
                out_dir=os.path.join("models", "best_model"))
            STATE["final_model"] = load_predictor(os.path.join("models", "best_model"))
            db = ResearchDB(agent.cfg.get("research_dir", "research"))
            generate_report(db, {**agent.summary(), "config": agent.cfg,
                                 "hardware": agent.hw},
                            os.path.join("models", "best_model", "research_report.md"))
    except Exception as e:
        STATE["last_error"] = f"final model: {e} {traceback.format_exc()}"


def _research_thread(cfg):
    try:
        agent = ResearchScientistAgent(cfg)
        STATE["agent"] = agent
        agent.research()
        _train_final_in_background(agent)
    except Exception as e:
        STATE["last_error"] = f"{e}\n{traceback.format_exc()}"


# ---------------------------------------------------------------- endpoints

@app.get("/api/status")
def status():
    agent = STATE["agent"]
    if agent is None:
        return {"status": "idle", "hardware": STATE["hardware"],
                "last_error": STATE["last_error"]}
    s = _agent_summary(agent)
    s["last_error"] = STATE["last_error"]
    s["final_model_ready"] = STATE["final_model"] is not None
    return s


@app.get("/api/dashboard")
def dashboard():
    agent = STATE["agent"]
    db = ResearchDB(agent.cfg.get("research_dir", "research")) if agent else ResearchDB()
    experiments = db.load_experiments()
    leaderboard = db.load_leaderboard()
    rows = {}
    for e in experiments:
        if e.get("score") is None:
            continue
        if e["name"] not in rows or e["score"] > rows[e["name"]]["score"]:
            rows[e["name"]] = {"name": e["name"], "score": e["score"],
                               "kind": e.get("kind"),
                               "ram_mb": e.get("metrics", {}).get("peak_rss_mb"),
                               "seconds": e.get("metrics", {}).get("train_seconds"),
                               "epochs": e.get("metrics", {}).get("epochs_completed"),
                               "generation": e.get("generation")}
    top = sorted(rows.values(), key=lambda r: -r["score"])[:20]
    if not leaderboard.get("rows"):
        db.save_leaderboard(top)
    out = {
        "status": "idle" if agent is None else agent.status,
        "hardware": STATE["hardware"],
        "experiments_count": len(experiments),
        "leaderboard": top,
        "journal": db.load_journal(limit=50),
        "lineage": db.load_lineage()["algorithms"],
        "final_model_ready": STATE["final_model"] is not None,
        "last_error": STATE["last_error"],
    }
    if agent:
        out["agent"] = _agent_summary(agent)
    return out


@app.get("/api/experiments")
def experiments():
    db = ResearchDB()
    out = []
    for e in db.load_experiments():
        out.append({k: v for k, v in e.items()
                    if k not in ("history", "genes")})
    return out


@app.get("/api/experiments/{exp_id}")
def experiment_detail(exp_id: int):
    db = ResearchDB()
    for e in db.load_experiments():
        if e.get("id") == exp_id:
            return e
    raise HTTPException(404, "experiment not found")


@app.get("/api/lineage")
def lineage():
    return ResearchDB().load_lineage()["algorithms"]


@app.get("/api/journal")
def journal(limit: int = 100):
    return ResearchDB().load_journal(limit=limit)


@app.get("/api/leaderboard")
def leaderboard():
    return ResearchDB().load_leaderboard()


@app.get("/api/hardware")
def hardware():
    return STATE["hardware"]


@app.get("/api/datasets")
def datasets():
    return {"builtins": list(BUILTIN_DATASETS.keys()),
            "note": "or provide a CSV path when starting research"}


@app.post("/api/research/start")
def research_start(req: ResearchStartReq):
    if STATE["thread"] and STATE["thread"].is_alive():
        raise HTTPException(409, "research already running")
    cfg = {
        "dataset": req.dataset,
        "research_seconds": req.seconds,
        "architecture": req.architecture,
        "epochs": req.epochs,
        "max_train_seconds": req.max_train_seconds,
        "population_size": req.population_size,
        "priority": req.priority,
        "seed": req.seed,
        "max_rows": req.max_rows,
        "hardware": STATE["hardware"],
        "research_dir": "research",
        "run_id": f"run_{int(time.time())}",
    }
    t = threading.Thread(target=_research_thread, args=(cfg,), daemon=True)
    STATE["thread"] = t
    STATE["last_error"] = None
    t.start()
    return {"started": True, "config": cfg}


@app.post("/api/research/stop")
def research_stop():
    agent = STATE["agent"]
    if agent is None:
        raise HTTPException(409, "no research running")
    agent.request_stop()
    return {"stopping": True}


@app.get("/api/model")
def model_info():
    if STATE["final_model"] is None:
        # try loading from disk if server restarted
        try:
            STATE["final_model"] = load_predictor(os.path.join("models", "best_model"))
        except Exception:
            raise HTTPException(404, "no trained model yet")
    meta, _ = STATE["final_model"]
    return meta


@app.post("/api/playground")
def playground(req: PlaygroundReq):
    if STATE["final_model"] is None:
        try:
            STATE["final_model"] = load_predictor(os.path.join("models", "best_model"))
        except Exception:
            raise HTTPException(404, "no trained model yet — run research first")
    meta, model = STATE["final_model"]
    try:
        return predict(meta, model, text=req.text, vector=req.vector)
    except Exception as e:
        raise HTTPException(400, f"prediction failed: {e}")


@app.get("/api/report")
def report():
    p = os.path.join("models", "best_model", "research_report.md")
    if os.path.exists(p):
        with open(p) as f:
            return {"markdown": f.read()}
    raise HTTPException(404, "no report yet — research must complete first")


@app.get("/api/health")
def health():
    return {"ok": True}
