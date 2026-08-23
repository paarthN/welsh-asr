"""Gradio demo for the fine-tuned Welsh ASR model.

Runs on a free HF Spaces CPU instance, where a 10s clip takes 15-20s to
transcribe, so the UI has to say so rather than look broken.
"""
import os

import gradio as gr
from transformers import pipeline

MODEL_ID = os.environ.get("MODEL_ID", "pnawani/welsh-asr-xlsr-300m")

pipe = pipeline("automatic-speech-recognition", model=MODEL_ID)


def transcribe(audio_path):
    if not audio_path:
        return "No audio received — record or upload a clip first."
    return pipe(audio_path)["text"]


demo = gr.Interface(
    fn=transcribe,
    inputs=gr.Audio(sources=["microphone", "upload"], type="filepath", label="Welsh audio"),
    outputs=gr.Textbox(label="Transcription", lines=3),
    title="Welsh Speech Recognition",
    description=(
        "Wav2Vec2 XLS-R (300m) fine-tuned on FLEURS Welsh. Speak Welsh or upload "
        "a clip. Running on a free CPU instance, so expect 15-20 seconds for a "
        "10 second clip."
    ),
    article=(
        "Character-level CTC model. It has no language model and no numeral "
        "handling, so spoken numbers are transcribed as words rather than digits."
    ),
    examples=[],  # populated with FLEURS test clips before deploying
    flagging_mode="never",
)

if __name__ == "__main__":
    demo.launch()
