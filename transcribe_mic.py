#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""单次录音 + 转录脚本：录制麦克风音频并将结果打入当前光标所在输入框。

用法：
    python3 transcribe_mic.py --duration 5
    python3 transcribe_mic.py --duration 10 --model iic/SenseVoiceSmall
    python3 transcribe_mic.py --duration 10 --model FunAudioLLM/Fun-ASR-Nano-2512 --language zh --hotwords "开放时间,菜单"
"""

import argparse
import subprocess
import sys
import time

import numpy as np

from model_adapter import build_auto_model, generate_text, select_device_for_model

SAMPLE_RATE = 16000


def paste_text(text: str, type_delay: float) -> None:
    backup = subprocess.run(["pbpaste"], capture_output=True).stdout
    subprocess.run(["pbcopy"], input=text.encode("utf-8"), check=True)

    time.sleep(0.05)
    time.sleep(type_delay)

    from pynput.keyboard import Controller as KeyboardController
    from pynput.keyboard import Key as KeyboardKey

    keyboard = KeyboardController()
    keyboard.press(KeyboardKey.cmd)
    keyboard.press("v")
    keyboard.release("v")
    keyboard.release(KeyboardKey.cmd)

    time.sleep(0.1)
    subprocess.run(["pbcopy"], input=backup, check=True)


def select_device(model_id: str, preferred_device: str | None = None) -> str:
    return select_device_for_model(model_id, preferred_device)


def record_audio(duration: float) -> np.ndarray:
    import sounddevice as sd

    print(f"Recording {duration}s — speak now...")
    frames = sd.rec(
        int(duration * SAMPLE_RATE),
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype="float32",
    )
    sd.wait()
    return frames.flatten()


def transcribe(
    model,
    model_id: str,
    audio_np: np.ndarray,
    language: str,
    hotwords: str,
) -> str:
    return generate_text(model, model_id, audio_np, language, hotwords)


def main():
    parser = argparse.ArgumentParser(description="录音并将转录结果打入光标所在输入框")
    parser.add_argument("--duration", type=float, default=5.0, help="录音时长（秒）")
    parser.add_argument("--model", default="FunAudioLLM/Fun-ASR-Nano-2512", help="模型 ID")
    parser.add_argument("--language", default="zh", help="识别语言提示，默认 zh")
    parser.add_argument("--hotwords", default="", help="逗号分隔的热词列表，仅 Fun-ASR-Nano 生效")
    parser.add_argument("--device", default="auto", help="推理设备：auto/cpu/mps/cuda")
    parser.add_argument("--type-delay", type=float, default=0.1, help="打字前延迟（秒）")
    args = parser.parse_args()

    preferred_device = None if args.device in {"", "auto"} else args.device
    device = select_device(args.model, preferred_device)
    print(f"Loading {args.model} on {device}...")

    model = build_auto_model(args.model, device=device, disable_update=True)

    print("Model ready.\n")
    print(">>> 请先点击目标输入框，然后等待录音开始 <<<")
    time.sleep(1.5)

    audio = record_audio(args.duration)
    print("Transcribing...")

    text = transcribe(model, args.model, audio, args.language, args.hotwords)
    print(f"Transcript: {text}")

    if not text.strip():
        print("(no speech detected)")
        return

    paste_text(text, args.type_delay)
    print("Done.")


if __name__ == "__main__":
    main()
