"""Kaggle launcher for the existing TinyGPT trainer.

Upload the repository (or its engine/ and data/ folders) to Kaggle, then run:
    !python /kaggle/working/codespaces-react/scripts/kaggle_train.py \
        --repo /kaggle/working/codespaces-react --hours 8

The current model is pure NumPy, so this script uses Kaggle CPU. It does not
pretend to use TPU/GPU. A JAX/PyTorch port is required for those accelerators.
Checkpoints and the final model are written under --output.
"""

import argparse
import os
import shutil
import sys
import time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", default="/kaggle/working/codespaces-react")
    parser.add_argument("--output", default="/kaggle/working/chat_model")
    parser.add_argument("--hours", type=float, default=8)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch", type=int, default=16)
    args = parser.parse_args()

    repo = os.path.abspath(args.repo)
    if not os.path.isdir(os.path.join(repo, "engine")):
        raise SystemExit(
            f"Repository not found at {repo}. Upload the project or set --repo.")
    sys.path.insert(0, repo)
    from engine import chat as C

    data = os.path.join(repo, "data", "dailydialog.parquet")
    if not os.path.exists(data):
        raise SystemExit(
            "Missing data/dailydialog.parquet. Upload the dataset to Kaggle "
            "or place it in the repository data/ directory.")

    os.makedirs(args.output, exist_ok=True)
    ckpt = os.path.join(args.output, "checkpoint")
    status = os.path.join(args.output, "status.txt")
    started = time.time()

    def write_status(text):
        with open(status, "w") as f:
            f.write(text + "\nupdated=" + time.strftime("%Y-%m-%d %H:%M:%S"))

    print("Loading DailyDialog and optional SODA data...", flush=True)
    dialogs = C.load_dialogues(path=data)
    texts = [utterance for dialogue in dialogs for utterance in dialogue]
    tok = C.Tokenizer().build(texts, min_count=3, max_vocab=16000)
    X, Y = C.make_samples(tok, dialogs, block=128)
    print(f"samples={X.shape[0]} vocab={tok.size}", flush=True)
    write_status(f"stage=ready\nsamples={X.shape[0]}\nsteps=0")

    resume = None
    try:
        opt_state = C.load_opt_state(ckpt)
        if opt_state is not None:
            model, checkpoint_tok, meta = C.load(ckpt)
            if (meta.get("d_model") == 512 and meta.get("n_layers") == 7
                    and meta.get("block") == 128
                    and meta.get("vocab_size") == tok.size):
                resume = (model, opt_state, meta.get("history") or [],
                          int(meta.get("steps") or 0))
                print(f"Resuming from step {resume[3]}", flush=True)
    except Exception as exc:
        print(f"No compatible checkpoint: {exc}", flush=True)

    def callback(entry):
        line = (f"stage={entry.get('stage', 'training')}\n"
                f"step={entry['step']}\nepoch={entry['epoch']}\n"
                f"loss={entry.get('loss')}\nelapsed_min={entry['elapsed'] / 60:.1f}")
        write_status(line)
        if entry.get("loss") is not None:
            print(line.replace("\n", " "), flush=True)

    model, history = C.train(
        tok, X, Y, d_model=512, n_layers=7, n_heads=8, block=128,
        epochs=args.epochs, batch=args.batch, lr=3e-4,
        max_minutes=args.hours * 60, callback=callback, resume=resume,
        ckpt_every_min=10, ckpt_dir=ckpt)
    C.save(model, tok, history, out_dir=args.output)
    write_status(f"stage=done\nsteps={history[-1]['step'] if history else 0}\n"
                 f"loss={history[-1]['loss'] if history else None}")
    print(f"Done. Model saved to {args.output}", flush=True)


if __name__ == "__main__":
    main()
