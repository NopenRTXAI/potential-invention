"""Train a fresh TinyGPT on a bounded FineWeb-Edu streaming sample.

This intentionally samples a finite number of remote documents instead of
pretending that an 800GB corpus can be consumed quickly on CPU. It keeps the
experiment reproducible and avoids storing the full dataset.
"""

import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from engine import chat as C
from engine.hf_stream import stream_texts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="HuggingFaceFW/fineweb-edu")
    parser.add_argument("--config", default="sample-10BT")
    parser.add_argument("--max-texts", type=int, default=10000)
    parser.add_argument("--minutes", type=float, default=240)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--output", default="models/fineweb_model")
    args = parser.parse_args()

    print(f"Streaming {args.dataset}/{args.config} ({args.max_texts} texts max)",
          flush=True)
    texts = []
    for batch in stream_texts(args.dataset, config=args.config,
                              max_texts=args.max_texts, batch_size=128):
        texts.extend(batch)
        if len(texts) % 1024 < len(batch):
            print(f"streamed {len(texts)} texts", flush=True)
    if not texts:
        raise SystemExit("No text received from the streaming dataset")

    # Keep the multilingual identity/programming curriculum in every run.
    curated = C.load_curated()
    texts.extend(text for dialogue in curated for text in dialogue)
    tok = C.Tokenizer().build(texts, min_count=2, max_vocab=16000)
    dialogues = [[text] for text in texts]
    X, Y = C.make_samples(tok, dialogues, block=128)
    print(f"samples={X.shape[0]} vocab={tok.size}", flush=True)

    def callback(entry):
        if entry.get("loss") is not None:
            print(f"step={entry['step']} epoch={entry['epoch']} "
                  f"loss={entry['loss']:.4f} "
                  f"elapsed_min={entry['elapsed'] / 60:.1f}", flush=True)

    model, history = C.train(
        tok, X, Y, d_model=512, n_layers=7, n_heads=8, block=128,
        epochs=args.epochs, batch=args.batch, lr=3e-4,
        max_minutes=args.minutes, callback=callback,
        ckpt_every_min=15, ckpt_dir=os.path.join(args.output, "checkpoint"))
    C.save(model, tok, history, out_dir=args.output)
    print(f"done; saved {args.output}", flush=True)


if __name__ == "__main__":
    main()
