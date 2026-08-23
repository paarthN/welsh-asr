"""Build the CTC vocabulary and Wav2Vec2 processor for FLEURS Welsh.

Vocab is built from the train split only (standard practice: the model must
not be fit to characters it can only have seen via val/test). Saves
vocab.json and a Wav2Vec2Processor to ./welsh-asr-processor for train.py to load.
"""
import json
from pathlib import Path

from transformers import Wav2Vec2CTCTokenizer, Wav2Vec2FeatureExtractor, Wav2Vec2Processor

from data import SAMPLE_RATE, load_split, normalize_text

PROCESSOR_DIR = Path(__file__).resolve().parent.parent / "welsh-asr-processor"


def build_vocab(train_texts):
    chars = set()
    for t in train_texts:
        chars.update(normalize_text(t))
    chars.discard(" ")

    vocab = {c: i for i, c in enumerate(sorted(chars))}
    vocab["|"] = len(vocab)  # word delimiter (standard Wav2Vec2 convention for space)
    vocab["[UNK]"] = len(vocab)
    vocab["[PAD]"] = len(vocab)
    return vocab


def main():
    train = load_split("train")
    vocab = build_vocab(train["transcription"])

    PROCESSOR_DIR.mkdir(exist_ok=True)
    vocab_path = PROCESSOR_DIR / "vocab.json"
    vocab_path.write_text(json.dumps(vocab, ensure_ascii=False, indent=2))
    print(f"vocab: {len(vocab)} tokens -> {vocab_path}")
    print("".join(k for k in vocab if k not in ("|", "[UNK]", "[PAD]")))

    tokenizer = Wav2Vec2CTCTokenizer(
        str(vocab_path), unk_token="[UNK]", pad_token="[PAD]", word_delimiter_token="|"
    )
    feature_extractor = Wav2Vec2FeatureExtractor(
        feature_size=1, sampling_rate=SAMPLE_RATE, padding_value=0.0,
        do_normalize=True, return_attention_mask=True,
    )
    processor = Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)
    processor.save_pretrained(str(PROCESSOR_DIR))
    print(f"processor saved -> {PROCESSOR_DIR}")


if __name__ == "__main__":
    main()
