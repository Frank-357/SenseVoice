#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""单次录音 + 转录脚本：录制麦克风音频并将结果打入当前光标所在输入框。

用法：
    python transcribe_mic.py --duration 5
    python transcribe_mic.py --duration 10 --model iic/SenseVoiceSmall
"""

import argparse
import sys
import time

import numpy as np
import sounddevice as sd
import torch

SAMPLE_RATE = 16000


def select_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def record_audio(duration: float) -> np.ndarray:
    print(f"Recording {duration}s — speak now...")
    frames = sd.rec(
        int(duration * SAMPLE_RATE),
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype="float32",
    )
    sd.wait()
    return frames.flatten()


def transcribe(model, kwargs, audio_np: np.ndarray) -> str:
    from funasr.utils.postprocess_utils import rich_transcription_postprocess

    res = model.inference(
        data_in=audio_np,
        language="auto",
        use_itn=True,
        ban_emo_unk=False,
        fs=SAMPLE_RATE,
        **kwargs,
    )
    return rich_transcription_postprocess(res[0][0]["text"])


def main():
    parser = argparse.ArgumentParser(description="录音并将转录结果打入光标所在输入框")
    parser.add_argument("--duration", type=float, default=5.0, help="录音时长（秒）")
    parser.add_argument("--model", default="iic/SenseVoiceSmall", help="模型 ID")
    parser.add_argument("--type-delay", type=float, default=0.1, help="打字前延迟（秒）")
    args = parser.parse_args()

    device = select_device()
    print(f"Loading {args.model} on {device}...")

    from model import SenseVoiceSmall

    model, kwargs = SenseVoiceSmall.from_pretrained(model=args.model, device=device)
    model.eval()

    print("Model ready.\n")
    print(">>> 请先点击目标输入框，然后等待录音开始 <<<")
    time.sleep(1.5)

    audio = record_audio(args.duration)
    print("Transcribing...")

    text = transcribe(model, kwargs, audio)
    print(f"Transcript: {text}")

    if not text.strip():
        print("(no speech detected)")
        return

    time.sleep(args.type_delay)
    from pynput.keyboard import Controller as KeyboardController

    KeyboardController().type(text)
    print("Done.")


if __name__ == "__main__":
    main()
