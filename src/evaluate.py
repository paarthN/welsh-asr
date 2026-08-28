"""Evaluate an ASR model on FLEURS Welsh and write per-utterance predictions.

Works for both the zero-shot Whisper baseline and the fine-tuned CTC model, so
that both are scored through the identical normalization and metric path.

    python src/evaluate.py --model openai/whisper-small --out results/whisper_baseline.json
"""
import argparse
import json
import time
from pathlib import Path

import jiwer
import torch
from transformers import pipeline

from data import SAMPLE_RATE, decode_audio, has_digit, load_split, normalize_text


def pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def score(preds, refs):
    return {
        "wer": jiwer.wer(refs, preds),
        "cer": jiwer.cer(refs, preds),
        "n": len(preds),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="openai/whisper-small")
    p.add_argument("--split", default="test")
    p.add_argument("--out", default="results/whisper_baseline.json")
    p.add_argument("--batch-size", type=int, default=8,
                   help="clips per inference batch")
    p.add_argument("--limit", type=int, default=0, help="smoke-test on N examples")
    p.add_argument("--progress-every", type=int, default=25)
    p.add_argument("--dtype", default="fp16", choices=["fp16", "fp32"])
    args = p.parse_args()

    device = pick_device()
    is_whisper = "whisper" in args.model.lower()
    print(f"model={args.model} device={device} whisper={is_whisper}", flush=True)

    kwargs = {}
    if is_whisper:
        # Test clips run to 49s; without chunking Whisper silently truncates at
        # 30s, which would unfairly inflate the baseline's WER.
        kwargs["chunk_length_s"] = 30
        kwargs["generate_kwargs"] = {"language": "cy", "task": "transcribe"}

    pipe = pipeline(
        "automatic-speech-recognition",
        model=args.model,
        device=device,
        dtype=torch.float16 if (args.dtype == "fp16" and device != "cpu") else torch.float32,
        **kwargs,
    )

    ds = load_split(args.split)
    if args.limit:
        ds = ds.select(range(args.limit))

    # Group similar-length clips so padding within a batch stays small, and
    # process in batches: one-at-a-time inference on MPS degrades badly as
    # allocations fragment over hundreds of iterations.
    order = sorted(range(len(ds)), key=lambda i: ds[i]["num_samples"])

    rows, started = {}, time.time()
    for start in range(0, len(order), args.batch_size):
        idx = order[start:start + args.batch_size]
        batch = [ds[i] for i in idx]
        audio = [decode_audio(ex) for ex in batch]
        texts = pipe([{"raw": a, "sampling_rate": SAMPLE_RATE} for a in audio],
                     batch_size=len(audio))
        for i, ex, a, t in zip(idx, batch, audio, texts):
            rows[i] = {
                "id": ex["id"],
                "duration_s": len(a) / SAMPLE_RATE,
                "pred": normalize_text(t["text"]),
                "ref": normalize_text(ex["transcription"]),
                "ref_raw": ex["transcription"],
            }
        done = len(rows)
        if device == "mps":
            torch.mps.empty_cache()
        if done % args.progress_every < args.batch_size:
            rate = done / (time.time() - started)
            print(f"  {done}/{len(ds)}  {rate:.2f} utt/s  "
                  f"eta {(len(ds)-done)/max(rate,1e-9)/60:.1f} min", flush=True)

    rows = [rows[i] for i in range(len(ds))]  # restore dataset order

    keep = [r for r in rows if r["ref"].strip()]
    clean = [r for r in keep if not has_digit(r["ref"])]
    results = {
        "model": args.model,
        "split": args.split,
        "device": device,
        "overall": score([r["pred"] for r in keep], [r["ref"] for r in keep]),
        "no_digits": score([r["pred"] for r in clean], [r["ref"] for r in clean]),
        "elapsed_s": round(time.time() - started, 1),
        "predictions": rows,
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, ensure_ascii=False, indent=2))

    o, c = results["overall"], results["no_digits"]
    print(f"\noverall   WER {o['wer']:.4f}  CER {o['cer']:.4f}  (n={o['n']})")
    print(f"no-digits WER {c['wer']:.4f}  CER {c['cer']:.4f}  (n={c['n']})")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
