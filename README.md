# Welsh ASR: fine-tuned XLS-R for Welsh speech recognition

[Model on the Hugging Face Hub](https://huggingface.co/pnawani/welsh-asr-xlsr-300m)

Wav2Vec2 XLS-R (300m) fine-tuned on FLEURS Welsh, scored against zero-shot
Whisper-small on the same test set and through the same normalization path.

## Results

FLEURS Welsh test, 1021 utterances.

| Model | WER ↓ | CER ↓ | WER, no digits | CER, no digits |
|---|---|---|---|---|
| Whisper-small (zero-shot) | 0.5987 | 0.2289 | 0.5853 | 0.2149 |
| **XLS-R 300m (this work)** | **0.3991** | **0.1141** | **0.3879** | **0.1098** |
| relative change | **−33.3%** | **−50.1%** | −33.7% | −48.9% |

Trained on 11.3 hours (2784 clips) for 2100 steps, roughly 12 epochs, on a free
Colab T4.

Both models are scored on identically normalized text: lowercased, apostrophes
removed, punctuation stripped, Welsh circumflex vowels kept. About a fifth of
utterances contain numerals, which a character-level CTC model cannot produce
from audio, so every figure appears both overall and on the digit-free subset.
The two models are always scored on the same subsets.

## Error analysis

### The gain is uniform across utterance length

![WER by utterance length](results/wer_by_length.png)

FLEURS Welsh utterances are long, with a median of 12.7 seconds, so the usual
0-3s/3-6s/6-10s buckets would put about 90% of this test set in one bar. With
buckets fitted to the corpus, the fine-tuned model wins in every bucket by a
similar margin. Neither model degrades on longer audio. Whisper does worst on
the shortest clips, where there is least context.

### Fine-tuning halved the errors without changing their character

![Character substitutions](results/confusion_matrix.png)

| | Whisper-small | XLS-R (fine-tuned) |
|---|---|---|
| Character substitutions | 14,032 | 6,681 (−52%) |
| Vowel-for-vowel confusions | 5,802 | 2,748 (−53%) |
| Vowel share of all substitutions | 41.3% | **41.1%** |

Fine-tuning cut substitutions roughly in half, but vowel-for-vowel confusion
still accounts for the same 41% of them. Welsh `y`, `u` and `i` are acoustically
close and map onto no single English vowel. The model got better at Welsh
without the difficulty moving elsewhere. Consonant errors cluster on the initial
mutation pairs (`c`/`g`, `t`/`d`, `p`/`b`), which fell by the same proportion,
817 to 387.

### Circumflex vowels are under-produced

The model emits 259 of the 457 circumflex vowels the references contain, or 57%.
`â` → `a` alone accounts for 147 substitutions, the largest error that is not a
vowel-for-vowel confusion.

The circumflex marks vowel length in Welsh (`tan` "fire" against `tân` "until"),
which is acoustically subtle and often recoverable only from context. A
character-level CTC model decodes each frame independently with no language
model, so it has no mechanism for that. A KenLM decoder over the CTC output is
the next thing to try and would probably recover much of this.

### Notable failure cases

Reference `yn union fel y maer lleuad yn tynnu ar y ddaear…`, prediction empty
(WER 1.00, 5.2s). The model emits nothing at all on 2 of 1021 clips, both
unusually short at 3.4s against a 14.2s median. Excluding them changes WER by
0.0013, so this is a curiosity rather than a real problem.

Reference `nid yw eu hymddygiad thermol mor sefydlog…`, prediction
`jdemalfhywio isnodaus difodaz laj cewzon…` (WER 0.98, 33.1s). On the longest
clips the output degenerates into strings that look orthographically Welsh but
mean nothing. CTC keeps emitting plausible character sequences with no lexical
constraint.

Reference `dechreuodd diwylliannau a llwythau hynafol…`, prediction
`aisiant gour goers en trifus fudant…` (WER 1.00, 12.3s). The same failure, and
a case where zero-shot Whisper instead produced the repetition loop
`ydych chin meddwl` eleven times over. The two architectures fail in different
ways.

## Setup

```bash
pip install -r requirements.txt

# Whisper-small baseline
python src/evaluate.py --model openai/whisper-small --out results/whisper_baseline.json

# fine-tuned model
python src/evaluate.py --model pnawani/welsh-asr-xlsr-300m --out results/finetuned_results.json

# comparison table, plots and failure cases
python src/error_analysis.py
```

Training runs from `notebooks/train_colab.ipynb`, or directly:

```bash
python src/prepare_data.py     # character vocabulary and processor
python src/train.py --max-steps 2100 --batch-size 2 --grad-accum 8
```

## Approach

### Data

FLEURS Welsh: 3427 train / 447 validation / 1021 test utterances, 12.2 hours of
training audio. Two filters run before training.

Clips over 30 seconds are dropped, which costs 1.7% of the data. The 20-second
cutoff used in most XLS-R tutorials would discard 31%, because FLEURS utterances
are unusually long.

The second filter matters more. About 15% of the train split pairs a truncated
clip with its full transcript, in the worst case 0.96 seconds of audio against
303 characters. wav2vec2 downsamples by 320, so CTC cannot emit those labels at
all and returns infinite loss, which becomes NaN gradients within two steps.
Those clips are short, so dropping 15% of them costs under 1 of 12.2 hours.

### Text

Lowercased, apostrophes deleted (`mae'r` → `maer`), punctuation stripped. The
Welsh circumflex vowels `âêîôŵŷ` are kept. Other diacritics fold onto their base
letters, since they appear only a handful of times in foreign names and would
otherwise add vocabulary entries with almost no training signal. That leaves a
45-token character vocabulary.

Both the baseline and the fine-tuned model run through this same function.
Inconsistency here is the usual source of misleading WER comparisons.

### Model

`facebook/wav2vec2-xls-r-300m` with a fresh CTC head, feature encoder frozen,
dropout disabled, `mask_time_prob=0.05`. Batch size 2 with gradient accumulation
8 for an effective batch of 16, learning rate 3e-4, 200 warmup steps, fp16.

### Constraints

Everything except the training loop runs on CPU: the baseline, preprocessing,
evaluation and analysis. That keeps the free Colab GPU for the one step that
needs it. Training survived four disconnections by checkpointing to Google Drive
and resuming.

## Repo structure

```
├── src/
│   ├── data.py            # dataset loading and text normalization
│   ├── prepare_data.py    # character vocabulary and processor
│   ├── train.py           # fine-tuning
│   ├── evaluate.py        # WER/CER for any model, one code path
│   └── error_analysis.py  # comparison table, plots, failure cases
├── notebooks/
│   ├── exploration.ipynb  # dataset statistics and the baseline
│   └── train_colab.ipynb  # training on a free Colab T4
├── app/app.py             # Gradio demo
└── results/               # metrics, plots, per-utterance predictions
```

## Acknowledgments

FLEURS (Google), XLS-R (Meta AI), Hugging Face Transformers.
