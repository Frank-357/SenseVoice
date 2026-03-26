#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""WhisperFlow-style floating overlay widget for hotkey daemon.

Uses tkinter (built-in) to display a pill-shaped, always-on-top, transparent
overlay at the bottom-center of the screen showing recording waveform,
processing spinner, and control buttons.
"""

import ctypes
import ctypes.util
import math
import tkinter as tk
from collections import deque
from typing import Callable, Optional


def _set_macos_accessory_app() -> None:
    """Set this process to NSApplicationActivationPolicyAccessory.

    This prevents the Python process from ever becoming the frontmost
    (active) app, so tkinter windows cannot steal focus from other apps.
    """
    try:
        objc = ctypes.cdll.LoadLibrary(ctypes.util.find_library("objc"))
        objc.objc_getClass.restype = ctypes.c_void_p
        objc.objc_getClass.argtypes = [ctypes.c_char_p]
        objc.sel_registerName.restype = ctypes.c_void_p
        objc.sel_registerName.argtypes = [ctypes.c_char_p]
        objc.objc_msgSend.restype = ctypes.c_void_p
        objc.objc_msgSend.argtypes = [ctypes.c_void_p, ctypes.c_void_p]

        NSApp = objc.objc_msgSend(
            objc.objc_getClass(b"NSApplication"),
            objc.sel_registerName(b"sharedApplication"),
        )

        # setActivationPolicy: takes NSInteger arg
        set_policy = ctypes.CFUNCTYPE(
            ctypes.c_bool, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_long
        )
        set_policy(objc.objc_msgSend)(
            NSApp,
            objc.sel_registerName(b"setActivationPolicy:"),
            1,  # NSApplicationActivationPolicyAccessory
        )
    except Exception:
        pass

# States (mirror hotkey_daemon constants)
IDLE = "IDLE"
HOLD_RECORDING = "HOLD_RECORDING"
PERSISTENT_RECORDING = "PERSISTENT_RECORDING"
TRANSCRIBING = "TRANSCRIBING"

# Layout
PILL_W, PILL_H = 180, 34
CORNER_R = 17
PAD_BOTTOM = 140  # px above screen bottom
BG_COLOR = "#1a1a2e"
BG_ALPHA = 0.85

# Waveform
WAVE_BARS = 16
BAR_WIDTH = 3
BAR_GAP = 2
BAR_MAX_H = 20
BAR_COLOR_HOLD = "#ff4d6d"
BAR_COLOR_PERSISTENT = "#4dc9f6"

# Spinner
SPINNER_R = 10
SPINNER_COLOR = "#4dc9f6"

# Buttons
BTN_R = 9
CANCEL_COLOR = "#ff4d6d"
STOP_COLOR = "#ff4d6d"


class OverlayWidget:
    """Floating pill overlay driven by state changes from daemon threads."""

    def __init__(
        self,
        on_cancel: Optional[Callable] = None,
        on_stop: Optional[Callable] = None,
    ):
        self._on_cancel = on_cancel
        self._on_stop = on_stop
        self._state = IDLE
        self._rms_values: deque[float] = deque([0.0] * WAVE_BARS, maxlen=WAVE_BARS)
        self._spinner_angle = 0.0

        # ── tkinter setup ──
        self._root = tk.Tk()
        _set_macos_accessory_app()  # must be after Tk() inits NSApplication
        self._root.overrideredirect(True)
        self._root.attributes("-topmost", True)
        # macOS transparency
        try:
            self._root.attributes("-transparent", True)
            self._root.config(bg="systemTransparent")
            self._use_transparent = True
        except tk.TclError:
            self._root.attributes("-alpha", BG_ALPHA)
            self._use_transparent = False
            self._root.config(bg=BG_COLOR)

        # Canvas
        self._canvas = tk.Canvas(
            self._root,
            width=PILL_W,
            height=PILL_H,
            highlightthickness=0,
            bd=0,
        )
        if self._use_transparent:
            self._canvas.config(bg="systemTransparent")
        else:
            self._canvas.config(bg=BG_COLOR)
        self._canvas.pack()

        # Compute on-screen and off-screen positions
        self._root.update_idletasks()
        sw = self._root.winfo_screenwidth()
        sh = self._root.winfo_screenheight()
        self._pos_visible = (
            (sw - PILL_W) // 2,
            sh - PILL_H - PAD_BOTTOM,
        )
        self._pos_hidden = (
            (sw - PILL_W) // 2,
            sh + 100,  # below screen edge, invisible
        )

        # Start off-screen (never use withdraw/deiconify)
        hx, hy = self._pos_hidden
        self._root.geometry(f"{PILL_W}x{PILL_H}+{hx}+{hy}")
        self._root.deiconify()  # ensure mapped once, then never withdraw

        # Bind custom event for thread-safe state updates
        self._root.bind("<<StateChange>>", self._on_state_change)
        self._root.bind("<<RMSPush>>", self._on_rms_update)

        self._pending_state: Optional[str] = None
        self._anim_id: Optional[str] = None

    # ── Public API (thread-safe) ──

    def set_state(self, state: str) -> None:
        """Called from any thread to update the overlay state."""
        self._pending_state = state
        try:
            self._root.event_generate("<<StateChange>>", when="tail")
        except tk.TclError:
            pass

    def push_audio_rms(self, rms: float) -> None:
        """Called from audio callback thread to feed waveform data."""
        self._rms_values.append(rms)
        try:
            self._root.event_generate("<<RMSPush>>", when="tail")
        except tk.TclError:
            pass

    def run(self) -> None:
        """Start the tkinter mainloop (call from main thread)."""
        self._root.mainloop()

    def stop(self) -> None:
        """Quit the mainloop."""
        try:
            self._root.quit()
        except tk.TclError:
            pass

    # ── Internal event handlers ──

    def _on_state_change(self, _event=None) -> None:
        new_state = self._pending_state
        if new_state is None or new_state == self._state:
            return
        self._state = new_state

        # Cancel any running animation
        if self._anim_id is not None:
            self._root.after_cancel(self._anim_id)
            self._anim_id = None

        if self._state == IDLE:
            self._hide()
        elif self._state == HOLD_RECORDING:
            self._show()
            self._draw_recording(persistent=False)
        elif self._state == PERSISTENT_RECORDING:
            self._show()
            self._draw_recording(persistent=True)
        elif self._state == TRANSCRIBING:
            self._show()
            self._spinner_angle = 0.0
            self._draw_spinner()

    def _on_rms_update(self, _event=None) -> None:
        if self._state in (HOLD_RECORDING, PERSISTENT_RECORDING):
            self._draw_recording(persistent=(self._state == PERSISTENT_RECORDING))

    # ── Show / Hide ──

    def _show(self) -> None:
        vx, vy = self._pos_visible
        self._root.geometry(f"{PILL_W}x{PILL_H}+{vx}+{vy}")

    def _hide(self) -> None:
        hx, hy = self._pos_hidden
        self._root.geometry(f"{PILL_W}x{PILL_H}+{hx}+{hy}")
        self._canvas.delete("all")

    # ── Drawing ──

    def _draw_pill(self) -> None:
        """Draw the pill background shape."""
        c = self._canvas
        c.delete("all")
        r = CORNER_R
        # Rounded rectangle via arcs + rectangles
        c.create_arc(0, 0, 2 * r, 2 * r, start=90, extent=90,
                      fill=BG_COLOR, outline=BG_COLOR)
        c.create_arc(PILL_W - 2 * r, 0, PILL_W, 2 * r, start=0, extent=90,
                      fill=BG_COLOR, outline=BG_COLOR)
        c.create_arc(0, PILL_H - 2 * r, 2 * r, PILL_H, start=180, extent=90,
                      fill=BG_COLOR, outline=BG_COLOR)
        c.create_arc(PILL_W - 2 * r, PILL_H - 2 * r, PILL_W, PILL_H,
                      start=270, extent=90, fill=BG_COLOR, outline=BG_COLOR)
        c.create_rectangle(r, 0, PILL_W - r, PILL_H,
                           fill=BG_COLOR, outline=BG_COLOR)
        c.create_rectangle(0, r, PILL_W, PILL_H - r,
                           fill=BG_COLOR, outline=BG_COLOR)

    def _draw_recording(self, persistent: bool = False) -> None:
        """Draw waveform bars and optional buttons."""
        self._draw_pill()
        c = self._canvas
        cy = PILL_H // 2
        bar_color = BAR_COLOR_PERSISTENT if persistent else BAR_COLOR_HOLD

        if persistent:
            # Cancel button (X) on the left
            bx, by = 20, cy
            c.create_oval(bx - BTN_R, by - BTN_R, bx + BTN_R, by + BTN_R,
                          fill=CANCEL_COLOR, outline=CANCEL_COLOR, tags="btn_cancel")
            # X mark
            d = 4
            c.create_line(bx - d, by - d, bx + d, by + d,
                          fill="white", width=2, tags="btn_cancel")
            c.create_line(bx - d, by + d, bx + d, by - d,
                          fill="white", width=2, tags="btn_cancel")
            c.tag_bind("btn_cancel", "<Button-1>", self._handle_cancel)

            # Stop button (square) on the right
            sx = PILL_W - 20
            c.create_oval(sx - BTN_R, by - BTN_R, sx + BTN_R, by + BTN_R,
                          fill=STOP_COLOR, outline=STOP_COLOR, tags="btn_stop")
            # Square icon
            sq = 4
            c.create_rectangle(sx - sq, by - sq, sx + sq, by + sq,
                               fill="white", outline="white", tags="btn_stop")
            c.tag_bind("btn_stop", "<Button-1>", self._handle_stop)

            wave_left = 36
            wave_right = PILL_W - 36
        else:
            # Red recording dot on the left
            dot_x, dot_y = 16, cy
            c.create_oval(dot_x - 5, dot_y - 5, dot_x + 5, dot_y + 5,
                          fill="#ff4d6d", outline="#ff4d6d")
            wave_left = 28
            wave_right = PILL_W - 10

        # Waveform bars
        avail_w = wave_right - wave_left
        total_bar_w = BAR_WIDTH + BAR_GAP
        n_bars = min(WAVE_BARS, avail_w // total_bar_w)
        start_x = wave_left + (avail_w - n_bars * total_bar_w) // 2

        rms_list = list(self._rms_values)
        # Normalize: find peak for scaling
        peak = max(max(rms_list), 0.01)

        for i in range(n_bars):
            idx = len(rms_list) - n_bars + i
            val = rms_list[idx] if 0 <= idx < len(rms_list) else 0.0
            h = max(3, int((val / peak) * BAR_MAX_H))
            x = start_x + i * total_bar_w
            c.create_rectangle(
                x, cy - h // 2, x + BAR_WIDTH, cy + h // 2,
                fill=bar_color, outline=bar_color,
            )

    def _draw_spinner(self) -> None:
        """Draw rotating arc spinner, schedule next frame."""
        self._draw_pill()
        c = self._canvas
        cx, cy = PILL_W // 2, PILL_H // 2
        r = SPINNER_R

        # "转录中..." label
        c.create_text(cx, cy - 1, text="转录中...", fill="#aaaaaa",
                      font=("Helvetica", 9))

        # Arc
        start = self._spinner_angle
        c.create_arc(
            cx - r, cy - r, cx + r, cy + r,
            start=start, extent=90,
            outline=SPINNER_COLOR, width=3, style="arc",
        )

        self._spinner_angle = (self._spinner_angle + 15) % 360
        self._anim_id = self._root.after(50, self._draw_spinner)

    # ── Button handlers ──

    def _handle_cancel(self, _event=None) -> None:
        if self._on_cancel:
            self._on_cancel()

    def _handle_stop(self, _event=None) -> None:
        if self._on_stop:
            self._on_stop()
