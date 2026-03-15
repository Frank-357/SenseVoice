#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""SenseVoice 全局快捷键语音转录守护进程。

快捷键：Left Cmd + Left Opt（同时按）

  长按模式：同时按住两键（听到 Pop 音）→ 说话 → 松开任意一键 → 转录 → 文字打入光标处
  双击模式：0.4s 内两次同时按下（听到 Tink 音）→ 说话 → 再次双击 → 转录 → 文字打入光标处

启动：
    python hotkey_daemon.py

macOS 权限：首次运行会弹出"输入监控"权限请求，在
    系统设置 → 隐私与安全性 → 输入监控 中允许 Terminal，然后重启脚本。
"""

import subprocess
import sys
import threading
import time

import numpy as np
import sounddevice as sd
import torch
from pynput import keyboard
from pynput.keyboard import Controller as KeyboardController
from pynput.keyboard import Key

# ── 配置区（可直接修改）────────────────────────────────────────────────────────

MODEL_DIR = "iic/SenseVoiceSmall"
# MODEL_DIR = "iic/speech_paraformer-large-vad-punc_asr_nat-zh-cn-16k-common-vocab8404-pytorch"

LANGUAGE = "auto"          # SenseVoice: "auto"/"zh"/"en"/"ja"  Paraformer: "zh"

DOUBLE_CLICK_THRESHOLD = 0.4   # 秒：两次组合键间隔 < 此值视为双击
HOLD_THRESHOLD = 0.15          # 秒：按下后持续此时间视为长按（而非单击）
TYPE_DELAY = 0.1               # 秒：转录完成后打字前延迟

SAMPLE_RATE = 16000

# ─────────────────────────────────────────────────────────────────────────────

# 状态常量
IDLE = "IDLE"
AWAITING_DECISION = "AWAITING_DECISION"
HOLD_RECORDING = "HOLD_RECORDING"
PERSISTENT_RECORDING = "PERSISTENT_RECORDING"
TRANSCRIBING = "TRANSCRIBING"

# ── 共享状态（均由 _state_lock 保护）─────────────────────────────────────────
_state_lock = threading.Lock()
_state = IDLE
_pressed_keys: set = set()
_last_combo_time: float = 0.0
_hold_timer: threading.Timer | None = None

# ── 录音状态（由 _rec_lock 保护）─────────────────────────────────────────────
_rec_lock = threading.Lock()
_recording = False
_audio_buffer: list[np.ndarray] = []

# ── 模型（主线程加载后只读）──────────────────────────────────────────────────
_model = None
_model_kwargs: dict = {}
_keyboard_ctrl = KeyboardController()

# ── 触发键集合────────────────────────────────────────────────────────────────
TRIGGER_KEYS: set = {Key.cmd_l, Key.alt_l}


# ── 工具函数 ──────────────────────────────────────────────────────────────────

def _notify(message: str) -> None:
    """macOS 通知（非阻塞）。"""
    subprocess.Popen(
        ["osascript", "-e", f'display notification "{message}" with title "SenseVoice"'],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def _play(sound: str, volume: float = 0.2) -> None:
    """播放 macOS 系统音效（非阻塞，默认音量 0.2）。"""
    subprocess.Popen(
        ["afplay", "-v", str(volume), f"/System/Library/Sounds/{sound}.aiff"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def _paste_text(text: str) -> None:
    """无痕剪贴板粘贴：备份剪贴板 → 写入文本 → Cmd+V → 恢复剪贴板。"""
    # Step a: 备份当前剪贴板（bytes，保留原始编码）
    backup = subprocess.run(["pbpaste"], capture_output=True).stdout

    # Step b: 将转录文本写入剪贴板
    subprocess.run(["pbcopy"], input=text.encode("utf-8"), check=True)

    # Step c: 短暂等待剪贴板写入生效
    time.sleep(0.05)

    # Step d: 模拟 Cmd+V 粘贴（绕过输入法）
    time.sleep(TYPE_DELAY)
    _keyboard_ctrl.press(Key.cmd)
    _keyboard_ctrl.press("v")
    _keyboard_ctrl.release("v")
    _keyboard_ctrl.release(Key.cmd)

    # Step e: 等待粘贴完成后恢复原始剪贴板
    time.sleep(0.1)
    subprocess.run(["pbcopy"], input=backup, check=True)


def _select_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


# ── 录音 ──────────────────────────────────────────────────────────────────────

_stream: sd.InputStream | None = None


def _audio_callback(indata: np.ndarray, frames: int, time_info, status) -> None:
    """sounddevice InputStream 回调，将帧追加到缓冲区。"""
    with _rec_lock:
        _audio_buffer.append(indata[:, 0].copy())


def _begin_recording() -> None:
    """开启麦克风流（此时 macOS 才显示橙点）。"""
    global _stream, _audio_buffer
    with _rec_lock:
        _audio_buffer = []
    _stream = sd.InputStream(
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype="float32",
        callback=_audio_callback,
        blocksize=1024,
    )
    _stream.start()


def _stop_recording() -> np.ndarray:
    """关闭麦克风流（橙点立刻消失）并返回录音数据。"""
    global _stream
    if _stream is not None:
        _stream.stop()
        _stream.close()
        _stream = None
    with _rec_lock:
        if _audio_buffer:
            return np.concatenate(_audio_buffer)
    return np.zeros(SAMPLE_RATE, dtype=np.float32)


# ── 推理 ──────────────────────────────────────────────────────────────────────

def _transcribe_and_type(audio_np: np.ndarray) -> None:
    """在独立线程中运行推理，完成后打字，最后将状态重置为 IDLE。"""
    global _state
    try:
        from funasr.utils.postprocess_utils import rich_transcription_postprocess

        res = _model.inference(
            data_in=audio_np,
            language=LANGUAGE,
            use_itn=True,
            ban_emo_unk=False,
            fs=SAMPLE_RATE,
            **_model_kwargs,
        )
        text = rich_transcription_postprocess(res[0][0]["text"])

        if text.strip():
            _paste_text(text)
            preview = text[:50] + ("..." if len(text) > 50 else "")
            _notify(f"✓ {preview}")
            print(f"[transcript] {text}")
        else:
            _notify("（无识别结果）")
            print("[transcript] (empty)")
    except Exception as exc:
        _notify(f"错误: {exc}")
        print(f"[error] {exc}", file=sys.stderr)
    finally:
        with _state_lock:
            _state = IDLE


# ── 定时器回调 ────────────────────────────────────────────────────────────────

def _hold_threshold_reached() -> None:
    """HOLD_THRESHOLD 秒后触发：若两键仍按下则进入长按录音模式。"""
    global _state
    with _state_lock:
        if _state != AWAITING_DECISION:
            return
        has_cmd = Key.cmd_l in _pressed_keys
        has_alt = any(k in _pressed_keys for k in (Key.alt_l, Key.alt))
        if has_cmd and has_alt:
            _state = HOLD_RECORDING
        else:
            _state = IDLE
            return

    # 在锁外执行 I/O
    _begin_recording()
    _play("Pop")
    _notify("录音中（松开停止）...")
    print("[state] HOLD_RECORDING")


# ── 核心：状态机事件处理 ──────────────────────────────────────────────────────

def _cancel_timer_locked() -> None:
    """取消 hold_timer，调用方必须持有 _state_lock。"""
    global _hold_timer
    if _hold_timer is not None:
        _hold_timer.cancel()
        _hold_timer = None


def _on_combo_pressed() -> None:
    """两键同时按下事件。"""
    global _state, _last_combo_time, _hold_timer

    now = time.time()
    action = None  # 锁外要执行的动作

    with _state_lock:
        interval = now - _last_combo_time

        if _state == IDLE:
            if interval < DOUBLE_CLICK_THRESHOLD:
                # 快速再次按下（从 IDLE 回来后）→ 双击
                _last_combo_time = now
                _state = PERSISTENT_RECORDING
                action = "start_persistent"
            else:
                # 第一次按下，等待判断长按/单击/双击
                _last_combo_time = now
                _state = AWAITING_DECISION
                _cancel_timer_locked()
                _hold_timer = threading.Timer(HOLD_THRESHOLD, _hold_threshold_reached)
                _hold_timer.daemon = True
                _hold_timer.start()

        elif _state == AWAITING_DECISION:
            # 第二次按下，且在 DOUBLE_CLICK_THRESHOLD 内 → 双击确认
            if interval < DOUBLE_CLICK_THRESHOLD:
                _last_combo_time = now
                _cancel_timer_locked()
                _state = PERSISTENT_RECORDING
                action = "start_persistent"
            # else: 忽略（不太可能：已经在等待中收到了新的 combo）

        elif _state == PERSISTENT_RECORDING:
            # 再次双击 → 停止持久录音
            if interval < DOUBLE_CLICK_THRESHOLD:
                _last_combo_time = now
                _state = TRANSCRIBING
                action = "stop_and_transcribe"
            else:
                _last_combo_time = now  # 更新时间，为下次双击做准备

        # HOLD_RECORDING / TRANSCRIBING：忽略

    # 锁外执行 I/O 操作
    if action == "start_persistent":
        _begin_recording()
        _play("Tink")
        _notify("持久录音中（再次双击停止）...")
        print("[state] PERSISTENT_RECORDING")
    elif action == "stop_and_transcribe":
        audio = _stop_recording()
        _notify("转录中...")
        print("[state] TRANSCRIBING")
        t = threading.Thread(target=_transcribe_and_type, args=(audio,), daemon=True)
        t.start()


def _on_trigger_key_released() -> None:
    """触发键（任一）松开事件。"""
    global _state
    action = None
    audio = None

    with _state_lock:
        if _state == AWAITING_DECISION:
            _cancel_timer_locked()
            _state = IDLE
            # 单击，忽略

        elif _state == HOLD_RECORDING:
            _state = TRANSCRIBING
            action = "stop_and_transcribe"

        # IDLE / PERSISTENT_RECORDING / TRANSCRIBING：忽略

    if action == "stop_and_transcribe":
        audio = _stop_recording()
        _notify("转录中...")
        print("[state] TRANSCRIBING")
        t = threading.Thread(target=_transcribe_and_type, args=(audio,), daemon=True)
        t.start()


# ── pynput 回调 ───────────────────────────────────────────────────────────────

def _is_alt(key) -> bool:
    return key in (Key.alt_l, Key.alt)


def on_press(key) -> None:
    try:
        if key == Key.cmd_l or _is_alt(key):
            # 动态发现实际的 alt 键名
            if _is_alt(key):
                TRIGGER_KEYS.add(key)

            with _state_lock:
                _pressed_keys.add(key)
                has_cmd = Key.cmd_l in _pressed_keys
                has_alt = any(_is_alt(k) for k in _pressed_keys)

            if has_cmd and has_alt:
                _on_combo_pressed()
    except Exception as e:
        print(f"[warn] on_press error (ignored): {e}", file=sys.stderr)


def on_release(key) -> None:
    try:
        if key == Key.cmd_l or _is_alt(key):
            was_in_combo = False
            with _state_lock:
                was_in_combo = (
                    Key.cmd_l in _pressed_keys
                    and any(_is_alt(k) for k in _pressed_keys)
                )
                _pressed_keys.discard(key)

            if was_in_combo:
                _on_trigger_key_released()
    except Exception as e:
        print(f"[warn] on_release error (ignored): {e}", file=sys.stderr)


# ── 主程序 ────────────────────────────────────────────────────────────────────

def main() -> None:
    global _model, _model_kwargs

    device = _select_device()
    print(f"Loading {MODEL_DIR} on {device} ...")

    from model import SenseVoiceSmall

    _model, _model_kwargs = SenseVoiceSmall.from_pretrained(model=MODEL_DIR, device=device)
    _model.eval()
    print("Model loaded.\n")

    _notify("SenseVoice 已就绪 (Left Cmd + Left Opt)")
    print("=" * 55)
    print("SenseVoice Hotkey Daemon — 已就绪")
    print("  长按模式：同时按住 Left Cmd + Left Opt（Pop 音）")
    print("            → 说话 → 松开任意键 → 文字自动打入")
    print("  双击模式：0.4s 内双击 Left Cmd + Left Opt（Tink 音）")
    print("            → 说话 → 再次双击 → 文字自动打入")
    print("按 Ctrl+C 退出")
    print("=" * 55)

    try:
        with keyboard.Listener(on_press=on_press, on_release=on_release) as listener:
            listener.join()
    except KeyboardInterrupt:
        pass

    if _stream is not None:
        _stream.stop()
        _stream.close()
    print("\nDaemon stopped.")


if __name__ == "__main__":
    main()
