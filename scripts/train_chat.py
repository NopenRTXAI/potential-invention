"""Standalone resumable TinyGPT trainer.

Run from the repository root:
    python scripts/train_chat.py --minutes 240

This process does not depend on the FastAPI server. It writes checkpoints to
models/chat_ckpt and a compact status file to models/chat_ckpt/status.json.
"""

import argparse
import json
import os
import signal
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from engine import chat as C

CKPT_DIR = os.path.join("models", "chat_ckpt")
STATUS = os.path.join(CKPT_DIR, "status.json")
STOP_REQUESTED = False


def write_status(**values):
    os.makedirs(CKPT_DIR, exist_ok=True)
    current = {}
    if os.path.exists(STATUS):
        try:
            with open(STATUS) as f:
                current = json.load(f)
        except (OSError, ValueError):
            pass
    current.update(values, updated=time.strftime("%Y-%m-%d %H:%M:%S"))
    tmp = STATUS + ".tmp"
    with open(tmp, "w") as f:
        json.dump(current, f)
    os.replace(tmp, STATUS)


def request_stop(signum, frame):
    global STOP_REQUESTED
    STOP_REQUESTED = True
    print(f"received signal {signum}; saving at the next checkpoint", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--minutes", type=float, default=240)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch", type=int, default=8)
    args = parser.parse_args()
    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)

    write_status(stage="loading", pid=os.getpid(), started=time.time())
    dialogs = C.load_dialogues()
    texts = [u for d in dialogs for u in d]
    tok = C.Tokenizer().build(texts, min_count=3, max_vocab=16000)
    X, Y = C.make_samples(tok, dialogs, block=128)
    write_status(stage="ready", samples=int(X.shape[0]), vocab=tok.size,
                 params=30325248, steps=0)

    resume = None
    try:
        opt_state = C.load_opt_state(CKPT_DIR)
        if opt_state is not None:
            model, ckpt_tok, meta = C.load(CKPT_DIR)
            if (meta.get("d_model") == 512 and meta.get("n_layers") == 7
                    and meta.get("block") == 128
                    and meta.get("vocab_size") == tok.size):
                resume = (model, opt_state, meta.get("history") or [],
                          int(meta.get("steps") or 0))
                print(f"resuming from step {resume[3]}", flush=True)
    except Exception as exc:
        print(f"checkpoint ignored: {exc}", flush=True)

    def callback(entry):
        write_status(stage=entry.get("stage", "training"),
                     steps=entry["step"], epoch=entry["epoch"],
                     loss=entry.get("loss"), elapsed_min=entry["elapsed"] / 60)

    model, hist = C.train(
        tok, X, Y, d_model=512, n_layers=7, n_heads=8, block=128,
        epochs=args.epochs, batch=args.batch, lr=3e-4, max_minutes=args.minutes,
        callback=callback, resume=resume, ckpt_every_min=15,
        ckpt_dir=CKPT_DIR)
    C.save(model, tok, hist)
    write_status(stage="done", steps=hist[-1]["step"] if hist else 0,
                 loss=hist[-1]["loss"] if hist else None,
                 epoch=hist[-1]["epoch"] if hist else None,
                 finished=time.time())
    print("training complete; final model saved", flush=True)


if __name__ == "__main__":
    main()
