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
  GET  /api/chat/status       chat model info (if trained)
  POST /api/chat              generate a real reply {message, history}
  POST /api/chat/train        train the chat model (background)
  GET  /api/weblearn/status   self-learning loop state + journal
  POST /api/weblearn/start    start autonomous web learning loop
  POST /api/weblearn/stop     stop the loop
  POST /api/weblearn/now      run one learn cycle immediately
  GET  /api/weblearn/search?q= web search (DuckDuckGo + Wikipedia)
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
    "chat_bot": None,          # engine.chat.ChatBot after training
    "chat_meta": None,         # metadata dict
    "chat_thread": None,       # training thread
    "chat_progress": {},       # live training progress
    "chat_error": None,
    "weblearn_loop": None,     # engine.weblearn.WebLearnLoop
    "weblearn_holder": {"bot": None, "meta": None},  # shared with loop
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


class ChatReq(BaseModel):
    message: str
    history: list[dict] = []   # [{speaker: "u"|"a", text: "..."}]


class ChatTrainReq(BaseModel):
    minutes: float = 240.0          # 4 hours default
    d_model: int = 512             # 30M params config
    n_layers: int = 7
    n_heads: int = 8
    block: int = 128
    vocab: int = 16000
    epochs: int = 2


class WebLearnReq(BaseModel):
    interval_min: float = 30.0     # learn every N minutes
    learn_minutes: float = 3.0     # fine-tune duration per cycle


class ChatSearchReq(BaseModel):
    message: str
    history: list[dict] = []
    use_web: bool = True           # include web results in reply


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


# ---------------------------------------------------------------- chat

def _chat_train_thread(req):
    """Train TinyGPT on DailyDialog+SODA in the background (30M config).

    Auto-resumes from the last checkpoint if one exists (server restart,
    terminal close, etc.) and saves a resumable checkpoint every 15 min.
    """
    try:
        from . import chat as C
        dialogs = C.load_dialogues()
        texts = [u for d in dialogs for u in d]
        tok = C.Tokenizer().build(texts, min_count=3, max_vocab=req.vocab)
        X, Y = C.make_samples(tok, dialogs, block=req.block)
        STATE["chat_progress"] = {"stage": "training", "samples": int(X.shape[0]),
                                  "vocab": tok.size, "steps": 0, "loss": None}

        def cb(entry):
            STATE["chat_progress"].update(
                steps=entry["step"], loss=round(entry["loss"], 4),
                epoch=entry["epoch"], elapsed_min=round(entry["elapsed"] / 60, 1))

        # auto-resume from checkpoint if present
        resume = None
        ckpt_dir = os.path.join("models", "chat_ckpt")
        try:
            opt_state = C.load_opt_state(out_dir=ckpt_dir)
            if opt_state is not None:
                model, tok_c, meta = C.load(out_dir=ckpt_dir)
                if (meta["d_model"] == req.d_model and
                        meta["n_layers"] == req.n_layers and
                        meta["block"] == req.block and
                        meta["vocab_size"] == tok.size):
                    hist = meta.get("history") or []
                    start_step = int(meta.get("steps") or 0)
                    resume = (model, opt_state, hist, start_step)
                    print(f"resuming 30M run from step {start_step}",
                          flush=True)
        except Exception as e:
            print(f"no checkpoint to resume ({e})", flush=True)

        model, hist = C.train(
            tok, X, Y, d_model=req.d_model, n_layers=req.n_layers,
            n_heads=req.n_heads, block=req.block, epochs=req.epochs,
            batch=32, lr=3e-4, max_minutes=req.minutes, callback=cb,
            resume=resume, ckpt_every_min=15, ckpt_dir=ckpt_dir)
        C.save(model, tok, hist)
        bot = C.ChatBot(model, tok, block=req.block)
        STATE["chat_bot"] = bot
        STATE["chat_meta"] = {
            "params": model.param_count(), "vocab": tok.size,
            "d_model": req.d_model, "n_layers": req.n_layers,
            "n_heads": req.n_heads, "block": req.block,
            "samples": int(X.shape[0]), "epochs": req.epochs,
            "final_loss": hist[-1]["loss"] if hist else None,
            "steps": hist[-1]["step"] if hist else 0,
        }
        STATE["chat_progress"] = {"stage": "done", **STATE["chat_meta"]}
        # register with the self-learning loop
        STATE["weblearn_holder"]["bot"] = bot
        STATE["weblearn_holder"]["meta"] = STATE["chat_meta"]
    except Exception as e:
        STATE["chat_error"] = f"{e}\n{traceback.format_exc()}"
        STATE["chat_progress"] = {"stage": "error", "error": str(e)}


@app.get("/api/chat/status")
def chat_status():
    if STATE["chat_bot"] is None:
        try:
            from . import chat as C
            model, tok, meta = C.load()
            STATE["chat_bot"] = C.ChatBot(model, tok, block=meta["block"])
            STATE["chat_meta"] = meta
        except Exception:
            pass
    return {
        "trained": STATE["chat_bot"] is not None,
        "training": STATE["chat_thread"] is not None
                    and STATE["chat_thread"].is_alive(),
        "progress": STATE["chat_progress"],
        "meta": STATE["chat_meta"],
        "error": STATE["chat_error"],
    }


@app.post("/api/chat")
def chat(req: ChatReq):
    if STATE["chat_bot"] is None:
        try:
            from . import chat as C
            model, tok, meta = C.load()
            STATE["chat_bot"] = C.ChatBot(model, tok, block=meta["block"])
            STATE["chat_meta"] = meta
            STATE["weblearn_holder"]["bot"] = STATE["chat_bot"]
            STATE["weblearn_holder"]["meta"] = meta
        except Exception:
            raise HTTPException(404, "chat model not trained yet — "
                                "POST /api/chat/train first")
    history = [(m.get("speaker", "u"), m.get("text", ""))
               for m in req.history][-6:]
    try:
        reply = STATE["chat_bot"].reply(history + [("u", req.message)])
        return {"reply": reply}
    except Exception as e:
        raise HTTPException(400, f"generation failed: {e}")


@app.post("/api/chat/smart")
def chat_smart(req: ChatSearchReq):
    """Chat with web search: model reply + real web facts with sources."""
    from . import weblearn as W
    web = None
    if req.use_web:
        try:
            web = W.web_answer(req.message)
        except Exception:
            web = None
    if STATE["chat_bot"] is None:
        try:
            from . import chat as C
            model, tok, meta = C.load()
            STATE["chat_bot"] = C.ChatBot(model, tok, block=meta["block"])
            STATE["chat_meta"] = meta
            STATE["weblearn_holder"]["bot"] = STATE["chat_bot"]
            STATE["weblearn_holder"]["meta"] = meta
        except Exception:
            pass  # model reply will be None; web answer may still exist
    reply = None
    if STATE["chat_bot"] is not None:
        try:
            history = [(m.get("speaker", "u"), m.get("text", ""))
                       for m in req.history][-6:]
            reply = STATE["chat_bot"].reply(history + [("u", req.message)])
        except Exception:
            reply = None
    if reply is None and web is None:
        raise HTTPException(404, "no chat model and no web answer — "
                            "train the model first")
    return {"reply": reply, "web": web}


@app.post("/api/chat/train")
def chat_train(req: ChatTrainReq):
    if STATE["chat_thread"] is not None and STATE["chat_thread"].is_alive():
        return {"started": False, "reason": "training already running"}
    STATE["chat_error"] = None
    STATE["chat_progress"] = {"stage": "loading"}
    t = threading.Thread(target=_chat_train_thread, args=(req,), daemon=True)
    STATE["chat_thread"] = t
    t.start()
    return {"started": True}


# ---------------------------------------------------------------- weblearn

@app.get("/api/weblearn/status")
def weblearn_status():
    from . import weblearn as W
    loop = STATE.get("weblearn_loop")
    running = loop is not None and loop.is_alive()
    return {
        "running": running,
        "loop_status": loop.status if running else "stopped",
        "interval_min": loop.interval_min if loop else None,
        "last_entry": loop.last_entry if loop else None,
        "stats": W.stats(),
        "journal": W.read_journal(30),
    }


@app.post("/api/weblearn/start")
def weblearn_start(req: WebLearnReq):
    from . import weblearn as W
    loop = STATE.get("weblearn_loop")
    if loop is not None and loop.is_alive():
        return {"started": False, "reason": "already running",
                "interval_min": loop.interval_min}
    # make sure holder has the current bot
    if STATE["weblearn_holder"]["bot"] is None and STATE["chat_bot"]:
        STATE["weblearn_holder"]["bot"] = STATE["chat_bot"]
        STATE["weblearn_holder"]["meta"] = STATE["chat_meta"]
    loop = W.WebLearnLoop(STATE["weblearn_holder"],
                          interval_min=req.interval_min,
                          learn_minutes=req.learn_minutes)
    STATE["weblearn_loop"] = loop
    loop.start()
    return {"started": True, "interval_min": req.interval_min,
            "learn_minutes": req.learn_minutes}


@app.post("/api/weblearn/stop")
def weblearn_stop():
    loop = STATE.get("weblearn_loop")
    if loop is None or not loop.is_alive():
        return {"stopped": False, "reason": "not running"}
    loop.stop_flag.set()
    return {"stopped": True}


@app.post("/api/weblearn/now")
def weblearn_now():
    """Run one learn cycle immediately (synchronous, ~1-4 min)."""
    from . import weblearn as W
    if STATE["weblearn_holder"]["bot"] is None and STATE["chat_bot"]:
        STATE["weblearn_holder"]["bot"] = STATE["chat_bot"]
        STATE["weblearn_holder"]["meta"] = STATE["chat_meta"]
    entry = W.learn_cycle(STATE["weblearn_holder"], minutes=2)
    return {"entry": entry, "stats": W.stats()}


@app.get("/api/weblearn/search")
def weblearn_search(q: str = ""):
    from . import weblearn as W
    if not q.strip():
        raise HTTPException(400, "missing ?q= parameter")
    try:
        return {"results": W.web_search(q)}
    except Exception as e:
        raise HTTPException(400, f"search failed: {e}")


@app.get("/api/health")
def health():
    return {"ok": True}
