#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import sys
from contextlib import contextmanager
import importlib.util
from pathlib import Path
from typing import Iterator

import numpy as np
import torch


FUN_ASR_NANO_MODELS = {
    "FunAudioLLM/Fun-ASR-Nano-2512",
    "FunAudioLLM/Fun-ASR-MLT-Nano-2512",
}

FUN_ASR_LANGUAGE_MAP = {
    "auto": None,
    "zh": "中文",
    "en": "英文",
    "ja": "日文",
    "中文": "中文",
    "英文": "英文",
    "日文": "日文",
    None: None,
    "": None,
}


def is_fun_asr_nano_model(model_id: str) -> bool:
    return model_id in FUN_ASR_NANO_MODELS


def is_huggingface_model(model_id: str) -> bool:
    return model_id.startswith("FunAudioLLM/")


def fun_asr_remote_root() -> Path:
    return Path(__file__).resolve().parent / "vendor" / "fun_asr_nano"


def fun_asr_remote_code() -> str:
    return str(fun_asr_remote_root() / "model.py")


@contextmanager
def prepend_sys_path(path: str) -> Iterator[None]:
    sys.path.insert(0, path)
    try:
        yield
    finally:
        if path in sys.path:
            sys.path.remove(path)


def build_auto_model(model_id: str, device: str, disable_update: bool = True):
    from funasr import AutoModel

    kwargs = {
        "model": model_id,
        "device": device,
        "disable_update": disable_update,
    }

    if is_fun_asr_nano_model(model_id):
        register_fun_asr_nano()
        kwargs.update(
            {
                "hub": "hf" if is_huggingface_model(model_id) else "ms",
            }
        )

    return AutoModel(**kwargs)


def select_device_for_model(model_id: str, preferred_device: str | None = None) -> str:
    if preferred_device:
        return preferred_device
    if torch.cuda.is_available():
        return "cuda"
    # Fun-ASR-Nano is slower on Apple MPS than CPU in this repo's current path.
    if is_fun_asr_nano_model(model_id) and sys.platform == "darwin":
        return "cpu"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def register_fun_asr_nano() -> None:
    from funasr.register import tables

    if "FunASRNano" in tables.model_classes:
        return

    remote_root = fun_asr_remote_root()
    module_name = "_fun_asr_nano_remote"
    module_path = remote_root / "model.py"

    with prepend_sys_path(str(remote_root)):
        spec = importlib.util.spec_from_file_location(module_name, module_path)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"无法加载 Fun-ASR remote code: {module_path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)


def normalize_fun_asr_language(language: str | None) -> str | None:
    return FUN_ASR_LANGUAGE_MAP.get(language, language)


def normalize_hotwords(hotwords: str | list[str] | None) -> list[str]:
    if hotwords is None:
        return []
    if isinstance(hotwords, str):
        return [item.strip() for item in hotwords.split(",") if item.strip()]
    return [item.strip() for item in hotwords if item.strip()]


def generate_text(
    model,
    model_id: str,
    audio_input,
    language: str | None,
    hotwords: str | list[str] | None = None,
):
    if is_fun_asr_nano_model(model_id):
        if isinstance(audio_input, np.ndarray):
            audio_input = torch.from_numpy(audio_input.astype(np.float32, copy=False))
        result = model.generate(
            input=[audio_input],
            cache={},
            batch_size=1,
            hotwords=normalize_hotwords(hotwords),
            language=normalize_fun_asr_language(language),
            itn=True,
        )
    else:
        result = model.generate(
            input=audio_input,
            cache={},
            language=language,
            use_itn=True,
            batch_size_s=60,
        )

    return result[0]["text"]
