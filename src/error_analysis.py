"""Error analysis over saved prediction JSON files.

Consumes the output of evaluate.py and produces the comparison table, the
WER-by-length plot, a character confusion matrix, and the worst failure cases.
Runs on CPU against saved JSON; no model or GPU needed.

    python src/error_analysis.py --baseline results/whisper_baseline.json \
                                 --finetuned results/finetuned_results.json
"""
import argparse
import collections
import json
from pathlib import Path

import jiwer
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# FLEURS Welsh clips are long (median ~13s); the usual 0-3/3-6/6-10/10+ buckets
# would put ~90% of this test set in a single bar.
BUCKETS = [(0, 8), (8, 12), (12, 16), (16, 20), (20, 10**6)]


def bucket_label(lo, hi):
    return f"{lo}-{hi}s" if hi < 10**6 else f"{lo}s+"


def load(path):
    return json.loads(Path(path).read_text())


def utt_wer(ref, pred):
    return jiwer.wer(ref, pred) if ref.strip() else float("nan")


def wer_by_length(runs, out_path):
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    width = 0.8 / len(runs)

    for k, (name, data) in enumerate(runs.items()):
        means = []
        for lo, hi in BUCKETS:
            vals = [
                utt_wer(r["ref"], r["pred"])
                for r in data["predictions"]
                if lo <= r["duration_s"] < hi and r["ref"].strip()
            ]
            means.append(np.nanmean(vals) if vals else np.nan)
        x = np.arange(len(BUCKETS)) + k * width
        ax.bar(x, means, width, label=name)
        for xi, m in zip(x, means):
            if not np.isnan(m):
                ax.text(xi, m + 0.01, f"{m:.2f}", ha="center", fontsize=8)

    ax.set_xticks(np.arange(len(BUCKETS)) + width * (len(runs) - 1) / 2)
    ax.set_xticklabels([bucket_label(lo, hi) for lo, hi in BUCKETS])
    ax.set_xlabel("utterance duration")
    ax.set_ylabel("mean WER")
    ax.set_title("WER by utterance length — FLEURS Welsh test")
    # Headroom so the legend cannot sit on top of a bar label.
    ax.set_ylim(0, np.nanmax([b.get_height() for b in ax.containers[0]]) * 1.28)
    ax.legend(loc="upper right", framealpha=0.95)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"wrote {out_path}")


def confusion_matrix(data, out_path, top_n=18):
    """Tally character substitutions from jiwer alignments."""
    refs = [r["ref"] for r in data["predictions"] if r["ref"].strip()]
    preds = [r["pred"] for r in data["predictions"] if r["ref"].strip()]
    out = jiwer.process_characters(refs, preds)

    subs = collections.Counter()
    for ref, hyp, chunks in zip(out.references, out.hypotheses, out.alignments):
        for c in chunks:
            if c.type != "substitute":
                continue
            for a, b in zip(ref[c.ref_start_idx:c.ref_end_idx],
                            hyp[c.hyp_start_idx:c.hyp_end_idx]):
                subs[(a, b)] += 1

    if not subs:
        print("no substitutions found")
        return

    chars = [c for c, _ in collections.Counter(
        {c: n for (a, b), n in subs.items() for c in (a, b)}
    ).most_common()]
    top = sorted(set(chars), key=lambda c: -sum(
        n for (a, b), n in subs.items() if c in (a, b)
    ))[:top_n]
    idx = {c: i for i, c in enumerate(top)}

    m = np.zeros((len(top), len(top)))
    for (a, b), n in subs.items():
        if a in idx and b in idx:
            m[idx[a], idx[b]] = n

    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    im = ax.imshow(m, cmap="magma")
    labels = ["␣" if c == " " else c for c in top]
    ax.set_xticks(range(len(top))); ax.set_xticklabels(labels)
    ax.set_yticks(range(len(top))); ax.set_yticklabels(labels)
    ax.set_xlabel("predicted"); ax.set_ylabel("reference")
    ax.set_title(f"Character substitutions — {data['model']}")
    fig.colorbar(im, ax=ax, label="count")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"wrote {out_path}")

    print("\ntop substitutions (reference -> predicted):")
    for (a, b), n in subs.most_common(12):
        print(f"  {a!r} -> {b!r}   {n}")


def comparison_table(runs):
    lines = ["| Model | WER | CER | WER (no digits) | CER (no digits) |",
             "|---|---|---|---|---|"]
    for name, d in runs.items():
        o, c = d["overall"], d["no_digits"]
        lines.append(f"| {name} | {o['wer']:.4f} | {o['cer']:.4f} | "
                     f"{c['wer']:.4f} | {c['cer']:.4f} |")
    if len(runs) == 2:
        (_, a), (_, b) = runs.items()
        rel = lambda x, y: (y - x) / x * 100
        lines.append(f"| **relative change** | {rel(a['overall']['wer'], b['overall']['wer']):+.1f}% | "
                     f"{rel(a['overall']['cer'], b['overall']['cer']):+.1f}% | "
                     f"{rel(a['no_digits']['wer'], b['no_digits']['wer']):+.1f}% | "
                     f"{rel(a['no_digits']['cer'], b['no_digits']['cer']):+.1f}% |")
    return "\n".join(lines)


def worst_cases(data, n=5):
    scored = [(utt_wer(r["ref"], r["pred"]), r)
              for r in data["predictions"] if r["ref"].strip()]
    # Very short references make WER trivially 1.0; require some length.
    scored = [(w, r) for w, r in scored if len(r["ref"].split()) >= 6]
    scored.sort(key=lambda t: -t[0])
    out = []
    for w, r in scored[:n]:
        out.append(f"- **WER {w:.2f}** ({r['duration_s']:.1f}s)\n"
                   f"  - Reference: `{r['ref'][:160]}`\n"
                   f"  - Predicted: `{r['pred'][:160]}`")
    return "\n".join(out)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--baseline", default="results/whisper_baseline.json")
    p.add_argument("--finetuned", default="results/finetuned_results.json")
    p.add_argument("--results-dir", default="results")
    args = p.parse_args()

    runs = {}
    runs["Whisper-small (zero-shot)"] = load(args.baseline)
    if args.finetuned and Path(args.finetuned).exists():
        runs["XLS-R 300m (fine-tuned)"] = load(args.finetuned)

    rd = Path(args.results_dir)
    rd.mkdir(parents=True, exist_ok=True)

    table = comparison_table(runs)
    print("\n" + table + "\n")

    wer_by_length(runs, rd / "wer_by_length.png")
    target = list(runs.values())[-1]
    confusion_matrix(target, rd / "confusion_matrix.png")

    cases = worst_cases(target)
    print("\nworst cases:\n" + cases)
    (rd / "error_analysis.md").write_text(
        f"# Error analysis\n\n## Comparison\n\n{table}\n\n"
        f"## Worst failure cases ({target['model']})\n\n{cases}\n"
    )
    print(f"\nwrote {rd / 'error_analysis.md'}")


if __name__ == "__main__":
    main()
