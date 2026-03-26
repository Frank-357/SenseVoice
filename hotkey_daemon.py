#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""全局快捷键语音转录守护进程。

快捷键：Left Cmd + Left Opt（同时按）

  长按模式：同时按住两键（听到 Pop 音）→ 说话 → 松开任意一键 → 转录 → 文字打入光标处
  双击模式：0.4s 内两次同时按下（听到 Tink 音）→ 说话 → 再次双击 → 转录 → 文字打入光标处

启动：
    python3 hotkey_daemon.py
    python3 hotkey_daemon.py --model FunAudioLLM/Fun-ASR-Nano-2512 --language zh --hotwords "开放时间,菜单"

macOS 权限：首次运行会弹出"输入监控"权限请求，在
    系统设置 → 隐私与安全性 → 输入监控 中允许 Terminal，然后重启脚本。
"""

import argparse
import atexit
import fcntl
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np

from model_adapter import build_auto_model, generate_text, select_device_for_model

# ── 配置区（可直接修改）────────────────────────────────────────────────────────

MODEL_DIR = os.getenv("SENSEVOICE_MODEL", "FunAudioLLM/Fun-ASR-Nano-2512")
LANGUAGE = os.getenv("SENSEVOICE_LANGUAGE", "zh")
HOTWORDS = os.getenv("SENSEVOICE_HOTWORDS", "")
DEVICE = os.getenv("SENSEVOICE_DEVICE", "")

# 可选模型：
#   iic/SenseVoiceSmall
#   FunAudioLLM/Fun-ASR-Nano-2512
# 可选语言：
#   SenseVoice: auto/zh/en/yue/ja/ko/nospeech
#   Fun-ASR-Nano: auto/zh/en/ja（内部会映射为 中文/英文/日文）

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
keyboard = None
KeyboardController = None
Key = None
_keyboard_ctrl = None

# ── 浮动窗口（主线程初始化，可能为 None）──────────────────────────────────────
_widget = None
_instance_lock_handle = None

# ── 触发键集合────────────────────────────────────────────────────────────────
TRIGGER_KEYS: set = set()


# ── 工具函数 ──────────────────────────────────────────────────────────────────

def _notify(message: str) -> None:
    """macOS 通知（非阻塞）。"""
    subprocess.Popen(
        ["osascript", "-e", f'display notification "{message}" with title "Paraformer"'],
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


def _load_pynput() -> None:
    global keyboard, KeyboardController, Key, _keyboard_ctrl, TRIGGER_KEYS

    if keyboard is not None:
        return

    from pynput import keyboard as keyboard_module
    from pynput.keyboard import Controller as keyboard_controller
    from pynput.keyboard import Key as key_class

    keyboard = keyboard_module
    KeyboardController = keyboard_controller
    Key = key_class
    _keyboard_ctrl = KeyboardController()
    TRIGGER_KEYS = {Key.cmd_l, Key.alt_l}


def _select_device(model_id: str, preferred_device: str | None = None) -> str:
    return select_device_for_model(model_id, preferred_device)


def _release_instance_lock() -> None:
    global _instance_lock_handle

    if _instance_lock_handle is None:
        return
    try:
        fcntl.flock(_instance_lock_handle.fileno(), fcntl.LOCK_UN)
    except OSError:
        pass
    try:
        _instance_lock_handle.close()
    except OSError:
        pass
    _instance_lock_handle = None


def _acquire_instance_lock() -> None:
    global _instance_lock_handle

    lock_dir = Path.home() / "Library" / "Caches" / "SenseVoice"
    lock_dir.mkdir(parents=True, exist_ok=True)
    lock_path = lock_dir / "hotkey_daemon.lock"

    handle = open(lock_path, "w")
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        handle.close()
        raise RuntimeError("hotkey_daemon.py 已经在运行，拒绝启动第二个实例。")

    handle.write(str(os.getpid()))
    handle.flush()
    _instance_lock_handle = handle
    atexit.register(_release_instance_lock)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="全局快捷键语音转录守护进程")
    parser.add_argument("--model", default=MODEL_DIR, help="模型 ID")
    parser.add_argument("--language", default=LANGUAGE, help="识别语言提示，默认 auto")
    parser.add_argument(
        "--hotwords",
        default=HOTWORDS,
        help="逗号分隔的热词列表，仅 Fun-ASR-Nano 生效",
    )
    parser.add_argument("--device", default=DEVICE, help="推理设备：auto/cpu/mps/cuda")
    parser.add_argument("--type-delay", type=float, default=TYPE_DELAY, help="粘贴前延迟（秒）")
    return parser.parse_args()


# ── 录音 ──────────────────────────────────────────────────────────────────────

_stream = None


def _audio_callback(indata: np.ndarray, frames: int, time_info, status) -> None:
    """sounddevice InputStream 回调，将帧追加到缓冲区。"""
    chunk = indata[:, 0].copy()
    with _rec_lock:
        _audio_buffer.append(chunk)
    if _widget:
        _widget.push_audio_rms(float(np.sqrt(np.mean(chunk ** 2))))


def _begin_recording() -> None:
    """开启麦克风流（此时 macOS 才显示橙点）。"""
    global _stream, _audio_buffer
    import sounddevice as sd

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
        text = generate_text(_model, MODEL_DIR, audio_np, LANGUAGE, HOTWORDS)

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
        if _widget:
            _widget.set_state(IDLE)


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
    if _widget:
        _widget.set_state(HOLD_RECORDING)
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
        if _widget:
            _widget.set_state(PERSISTENT_RECORDING)
        print("[state] PERSISTENT_RECORDING")
    elif action == "stop_and_transcribe":
        audio = _stop_recording()
        _notify("转录中...")
        if _widget:
            _widget.set_state(TRANSCRIBING)
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
        if _widget:
            _widget.set_state(TRANSCRIBING)
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


# ── 悬浮窗按钮回调 ────────────────────────────────────────────────────────────

def _cancel_persistent_recording() -> None:
    """悬浮窗 X 按钮：取消持久录音，丢弃音频。"""
    global _state
    with _state_lock:
        if _state != PERSISTENT_RECORDING:
            return
        _state = IDLE
    _stop_recording()  # 丢弃音频
    _play("Funk", volume=0.15)
    if _widget:
        _widget.set_state(IDLE)
    print("[state] IDLE (cancelled)")


def _stop_persistent_recording() -> None:
    """悬浮窗 Stop 按钮：停止持久录音，开始转录。"""
    global _state
    with _state_lock:
        if _state != PERSISTENT_RECORDING:
            return
        _state = TRANSCRIBING
    audio = _stop_recording()
    _notify("转录中...")
    if _widget:
        _widget.set_state(TRANSCRIBING)
    print("[state] TRANSCRIBING")
    t = threading.Thread(target=_transcribe_and_type, args=(audio,), daemon=True)
    t.start()


# ── 主程序 ────────────────────────────────────────────────────────────────────

def main() -> None:
    global _model, _widget, MODEL_DIR, LANGUAGE, HOTWORDS, TYPE_DELAY, DEVICE

    args = _parse_args()
    MODEL_DIR = args.model
    LANGUAGE = args.language
    HOTWORDS = args.hotwords
    DEVICE = args.device
    TYPE_DELAY = args.type_delay

    _acquire_instance_lock()
    _load_pynput()
    preferred_device = None if DEVICE in {"", "auto"} else DEVICE
    device = _select_device(MODEL_DIR, preferred_device)
    print(f"Loading {MODEL_DIR} on {device} ...")

    _model = build_auto_model(MODEL_DIR, device=device, disable_update=True)
    print("Model loaded.\n")

    # 初始化悬浮窗（失败则回退到纯通知模式）
    try:
        from overlay_widget import OverlayWidget
        _widget = OverlayWidget(
            on_cancel=_cancel_persistent_recording,
            on_stop=_stop_persistent_recording,
        )
        print("[overlay] Widget initialized")
    except Exception as exc:
        _widget = None
        print(f"[overlay] Failed to init widget, notification-only mode: {exc}")

    _notify("Paraformer 已就绪 (Left Cmd + Left Opt)")
    print("=" * 55)
    print("Paraformer Hotkey Daemon — 已就绪")
    print(f"  模型：{MODEL_DIR}")
    print(f"  设备：{device}")
    print(f"  语言：{LANGUAGE}")
    if HOTWORDS.strip():
        print(f"  热词：{HOTWORDS}")
    print("  长按模式：同时按住 Left Cmd + Left Opt（Pop 音）")
    print("            → 说话 → 松开任意键 → 文字自动打入")
    print("  双击模式：0.4s 内双击 Left Cmd + Left Opt（Tink 音）")
    print("            → 说话 → 再次双击 → 文字自动打入")
    print("按 Ctrl+C 退出")
    print("=" * 55)

    # pynput listener 作为守护线程运行
    listener = keyboard.Listener(on_press=on_press, on_release=on_release)
    listener.daemon = True
    listener.start()

    try:
        if _widget:
            # 主线程运行 tkinter mainloop
            _widget.run()
        else:
            # 无窗口模式：主线程等待 listener
            listener.join()
    except KeyboardInterrupt:
        pass

    if _widget:
        _widget.stop()
    if _stream is not None:
        _stream.stop()
        _stream.close()
    print("\nDaemon stopped.")


if __name__ == "__main__":
    main()
