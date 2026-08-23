"""Fine-tune XLS-R-300m for Welsh CTC ASR on FLEURS.

Designed for a free Colab T4: audio is decoded and featurized inside the data
collator, so nothing is cached to disk and a reconnect costs only a dataset
re-download. Length grouping reads the FLEURS `num_samples` column directly,
which avoids a preprocessing pass just to learn clip lengths.

    python src/train.py --max-steps 2100 --batch-size 2 --grad-accum 8
"""
import argparse
from dataclasses import dataclass
from pathlib import Path

import jiwer
import numpy as np
import torch
from transformers import (
    Trainer,
    TrainingArguments,
    Wav2Vec2ForCTC,
    Wav2Vec2Processor,
)

from data import SAMPLE_RATE, decode_audio, load_split, normalize_text

BASE_MODEL = "facebook/wav2vec2-xls-r-300m"
# The wav2vec2 conv feature extractor downsamples the waveform by this factor.
CONV_STRIDE = 320
PROCESSOR_DIR = Path(__file__).resolve().parent.parent / "welsh-asr-processor"


@dataclass
class DataCollatorCTCWithPadding:
    """Decode, featurize, tokenize and pad a batch.

    Doing this here rather than in a dataset .map() keeps the raw columns intact
    (so length grouping can read `num_samples`) and avoids caching a multi-GB
    copy of the featurized audio.
    """

    processor: Wav2Vec2Processor

    def __call__(self, features):
        audio = [decode_audio(f) for f in features]
        batch = self.processor.feature_extractor(
            audio, sampling_rate=SAMPLE_RATE, padding=True, return_tensors="pt",
            return_attention_mask=True,
        )

        texts = [normalize_text(f["transcription"]) for f in features]
        labels_batch = self.processor.tokenizer(
            texts, padding=True, return_tensors="pt",
        )
        # -100 is ignored by the CTC loss.
        batch["labels"] = labels_batch["input_ids"].masked_fill(
            labels_batch["attention_mask"].ne(1), -100
        )
        return batch


def build_compute_metrics(processor):
    pad_id = processor.tokenizer.pad_token_id

    def compute_metrics(pred):
        # preprocess_logits_for_metrics already reduced logits to ids.
        pred_ids = pred.predictions
        label_ids = np.where(pred.label_ids == -100, pad_id, pred.label_ids)

        pred_str = processor.batch_decode(pred_ids)
        label_str = processor.batch_decode(label_ids, group_tokens=False)

        pairs = [(p, l) for p, l in zip(pred_str, label_str) if l.strip()]
        if not pairs:
            return {"wer": 1.0, "cer": 1.0}
        preds, refs = [p for p, _ in pairs], [l for _, l in pairs]
        return {"wer": jiwer.wer(refs, preds), "cer": jiwer.cer(refs, preds)}

    return compute_metrics


def preprocess_logits_for_metrics(logits, labels):
    """Collapse logits to predicted ids on-device.

    Without this the Trainer accumulates full (batch, frames, vocab) float
    logits for the whole eval set, which is a needless memory spike.
    """
    if isinstance(logits, tuple):
        logits = logits[0]
    return logits.argmax(dim=-1)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output-dir", default="./welsh-asr-xlsr")
    p.add_argument("--max-steps", type=int, default=2100)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--grad-accum", type=int, default=8)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--warmup-steps", type=int, default=200)
    p.add_argument("--save-steps", type=int, default=400)
    p.add_argument("--eval-steps", type=int, default=400)
    p.add_argument("--max-duration", type=float, default=30.0)
    p.add_argument("--ctc-margin", type=float, default=2.0,
                   help="require frames >= margin * label chars")
    p.add_argument("--eval-max-samples", type=int, default=0, help="subsample val for speed")
    p.add_argument("--gradient-checkpointing", action="store_true")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--push-to-hub", action="store_true")
    p.add_argument("--hub-model-id", default=None)
    p.add_argument("--run-name", default="welsh-xlsr-run1")
    p.add_argument("--smoke", action="store_true", help="tiny CPU run to validate plumbing")
    args = p.parse_args()

    processor = Wav2Vec2Processor.from_pretrained(str(PROCESSOR_DIR))

    max_samples = int(args.max_duration * SAMPLE_RATE)

    def keep(example):
        """Drop clips that are too long, and clips CTC cannot possibly emit.

        ~15% of the FLEURS Welsh train split pairs a truncated clip with its
        full transcript (worst case: 0.96s of audio for 303 characters). CTC
        returns inf for those, which turns into NaN gradients and wrecks the
        run. They are short, so dropping them costs 15% of the clips but under
        1 of 12.2 audio hours.
        """
        if example["num_samples"] > max_samples:
            return False
        frames = example["num_samples"] // CONV_STRIDE
        return frames >= args.ctc_margin * len(normalize_text(example["transcription"]))

    train = load_split("train").filter(keep)
    val = load_split("validation").filter(keep)
    if args.eval_max_samples:
        val = val.select(range(min(args.eval_max_samples, len(val))))
    if args.smoke:
        train, val = train.select(range(4)), val.select(range(2))
    print(f"train={len(train)}  val={len(val)}  vocab={len(processor.tokenizer)}", flush=True)

    model = Wav2Vec2ForCTC.from_pretrained(
        BASE_MODEL,
        attention_dropout=0.0,
        hidden_dropout=0.0,
        feat_proj_dropout=0.0,
        mask_time_prob=0.05,
        layerdrop=0.0,
        ctc_loss_reduction="mean",
        # Safety net: any infeasible sample that survives filtering contributes
        # zero instead of inf, rather than poisoning the whole batch with NaN.
        ctc_zero_infinity=True,
        pad_token_id=processor.tokenizer.pad_token_id,
        vocab_size=len(processor.tokenizer),
    )
    model.freeze_feature_encoder()

    use_cuda = torch.cuda.is_available()
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        # transformers 5.x: replaces the old group_by_length=True flag.
        train_sampling_strategy="group_by_length",
        length_column_name="num_samples",
        # The collator needs the raw audio/transcription columns, which the
        # Trainer would otherwise strip as "unused".
        remove_unused_columns=False,
        per_device_train_batch_size=1 if args.smoke else args.batch_size,
        per_device_eval_batch_size=1 if args.smoke else args.batch_size,
        gradient_accumulation_steps=1 if args.smoke else args.grad_accum,
        max_steps=2 if args.smoke else args.max_steps,
        learning_rate=args.lr,
        warmup_steps=0 if args.smoke else args.warmup_steps,
        eval_strategy="steps",
        eval_steps=2 if args.smoke else args.eval_steps,
        save_steps=args.save_steps,
        save_total_limit=2,
        logging_steps=1 if args.smoke else 100,
        fp16=use_cuda and not args.smoke,
        gradient_checkpointing=args.gradient_checkpointing,
        use_cpu=args.smoke,
        dataloader_num_workers=0 if args.smoke else 2,
        report_to="none" if args.smoke else "wandb",
        run_name=args.run_name,
        # Deliberately NOT push_to_hub=True: that uploads ~1.2GB at every
        # save_steps. Checkpoints live on Drive; only the final model is pushed.
        push_to_hub=False,
        hub_model_id=args.hub_model_id,
        load_best_model_at_end=not args.smoke,
        metric_for_best_model="wer",
        greater_is_better=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=DataCollatorCTCWithPadding(processor=processor),
        compute_metrics=build_compute_metrics(processor),
        preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        train_dataset=train,
        eval_dataset=val,
        processing_class=processor,
    )

    trainer.train(resume_from_checkpoint=args.resume or None)
    trainer.save_model(args.output_dir)
    processor.save_pretrained(args.output_dir)
    if args.push_to_hub:
        trainer.push_to_hub()


if __name__ == "__main__":
    main()
