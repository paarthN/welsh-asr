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
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--limit", type=int, default=0, help="smoke-test on N examples")
    args = p.parse_args()

    device = pick_device()
    is_whisper = "whisper" in args.model.lower()
    print(f"model={args.model} device={device} whisper={is_whisper}")

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
        dtype=torch.float16 if device != "cpu" else torch.float32,
        **kwargs,
    )

    ds = load_split(args.split)
    if args.limit:
        ds = ds.select(range(args.limit))

    rows, started = [], time.time()
    for i, ex in enumerate(ds):
        audio = decode_audio(ex)
        text = pipe({"raw": audio, "sampling_rate": SAMPLE_RATE})["text"]
        rows.append({
            "id": ex["id"],
            "duration_s": len(audio) / SAMPLE_RATE,
            "pred": normalize_text(text),
            "ref": normalize_text(ex["transcription"]),
            "ref_raw": ex["transcription"],
        })
        if (i + 1) % 50 == 0:
            rate = (i + 1) / (time.time() - started)
            print(f"  {i+1}/{len(ds)}  {rate:.2f} utt/s  eta {(len(ds)-i-1)/rate/60:.1f} min")

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
