"""Loading and text normalization for FLEURS Welsh.

Both the baseline and the fine-tuned model must score against identically
normalized text, so every consumer goes through normalize_text() here.
"""
import io
import re
import unicodedata

import numpy as np
import soundfile as sf
from datasets import Audio, load_dataset

SAMPLE_RATE = 16000

# Circumflex vowels are standard Welsh orthography (gwr, ty) and are kept.
# Other diacritics come from foreign names and are folded to their base letter
# so they do not add near-untrainable tokens to the vocabulary.
WELSH_DIACRITICS = set("âêîôûŵŷ")
ALLOWED = set("abcdefghijklmnopqrstuvwxyz0123456789 ") | WELSH_DIACRITICS

_APOSTROPHE = re.compile("['’ʼ‘]")
_WHITESPACE = re.compile(r"\s+")
_DIGIT = re.compile(r"\d")


def normalize_text(text: str) -> str:
    """Lowercase, drop apostrophes and punctuation, keep Welsh circumflex vowels."""
    text = unicodedata.normalize("NFC", text).lower()
    # Delete apostrophes rather than space them: mae'r -> maer, not mae r.
    text = _APOSTROPHE.sub("", text)

    out = []
    for ch in text:
        if ch in ALLOWED:
            out.append(ch)
        elif ch.isspace():
            out.append(" ")
        else:
            # Fold e/c to e/c; anything with no plain-ASCII base is dropped.
            base = "".join(
                c for c in unicodedata.normalize("NFD", ch)
                if not unicodedata.combining(c)
            ).lower()
            out.append(base if base and base in ALLOWED else " ")

    return _WHITESPACE.sub(" ", "".join(out)).strip()


def has_digit(text: str) -> bool:
    return bool(_DIGIT.search(text))


def load_split(split: str):
    """FLEURS Welsh with audio left undecoded (no torchcodec/ffmpeg needed)."""
    return load_dataset("google/fleurs", "cy_gb", split=split).cast_column(
        "audio", Audio(decode=False)
    )


def decode_audio(example) -> np.ndarray:
    """Decode one example's WAV bytes to a mono float32 array at 16 kHz."""
    array, sr = sf.read(io.BytesIO(example["audio"]["bytes"]), dtype="float32")
    if sr != SAMPLE_RATE:
        raise ValueError(f"expected {SAMPLE_RATE} Hz, got {sr}")
    if array.ndim > 1:
        array = array.mean(axis=1)
    return array


if __name__ == "__main__":
    assert normalize_text("Mae'r term saffari") == "maer term saffari"
    assert normalize_text("Y TŶ GWYN, 2019!") == "y tŷ gwyn 2019"
    assert normalize_text("café  —  ﻿test") == "cafe test"
    assert normalize_text("15% o £30") == "15 o 30"
    assert normalize_text("gŵr â ŷ ô") == "gŵr â ŷ ô"
    assert has_digit("2019") and not has_digit("dwy fil")
    print("ok")
