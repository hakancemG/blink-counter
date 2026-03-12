"""
Göz Kırpma Sayacı - GUI
Tkinter + Pillow tabanlı modern dark-theme arayüz.
Çalıştırmak için: python gui.py
Bağımlılıklar: pip install opencv-python mediapipe pillow
"""

import tkinter as tk
from tkinter import font as tkfont
import threading
import time
import math
import platform
from PIL import Image, ImageTk
import cv2

from blink_counter import BlinkDetector, LOW_BLINK_THRESHOLD

# ── Renk paleti ────────────────────────────────────────────────────────
BG          = "#0d0f14"
PANEL       = "#13161e"
PANEL_LIGHT = "#1a1d28"
BORDER      = "#22263a"
ACCENT      = "#4f8ef7"
GREEN       = "#34d399"
AMBER       = "#fbbf24"
RED_COL     = "#f87171"
TEXT_PRI    = "#e8eaf6"
TEXT_SEC    = "#6b7280"
TEXT_DIM    = "#2e3248"

BAR_W, BAR_H = 340, 90


def hex_to_rgb(h):
    h = h.lstrip("#")
    return tuple(int(h[i:i+2], 16) for i in (0, 2, 4))


def lerp_color(c1, c2, t):
    t = max(0.0, min(1.0, t))
    return "#{:02x}{:02x}{:02x}".format(
        int(c1[0] + (c2[0]-c1[0])*t),
        int(c1[1] + (c2[1]-c1[1])*t),
        int(c1[2] + (c2[2]-c1[2])*t),
    )


# ── Widgets ────────────────────────────────────────────────────────────

class RingGauge(tk.Canvas):
    """Dairesel EAR göstergesi."""

    def __init__(self, parent, size=120, **kw):
        super().__init__(parent, width=size, height=size,
                         bg=BG, highlightthickness=0, **kw)
        self.size = size
        self._draw(0.0)

    def set(self, value: float):
        self._draw(max(0.0, min(1.0, value)))

    def _draw(self, v):
        self.delete("all")
        s, pad = self.size, 10
        x0, y0, x1, y1 = pad, pad, s-pad, s-pad
        cx, cy = s/2, s/2

        self.create_arc(x0, y0, x1, y1, start=90, extent=360,
                        outline=PANEL_LIGHT, width=8, style="arc")
        if v > 0:
            color = lerp_color(hex_to_rgb(RED_COL), hex_to_rgb(GREEN), v)
            self.create_arc(x0, y0, x1, y1, start=90, extent=-v*360,
                            outline=color, width=8, style="arc")

        self.create_text(cx, cy-8, text=f"{int(v*100)}",
                         fill=TEXT_PRI, font=("Courier New", 18, "bold"))
        self.create_text(cx, cy+10, text="EAR %",
                         fill=TEXT_SEC, font=("Courier New", 7))


class BarChart(tk.Canvas):
    """Dakika bazlı kırpma geçmişi."""

    def __init__(self, parent, **kw):
        super().__init__(parent, width=BAR_W, height=BAR_H,
                         bg=PANEL, highlightthickness=0, **kw)
        self._draw([])

    def update_bars(self, data: list):
        self._draw(data)

    def _draw(self, data):
        self.delete("all")
        if not data:
            self.create_text(BAR_W//2, BAR_H//2,
                             text="Dakika geçmişi burada görünecek",
                             fill=TEXT_DIM, font=("Courier New", 8))
            return

        w, h  = BAR_W, BAR_H
        pad_x, pad_y = 16, 10
        n     = max(len(data), 10)
        max_v = max(max(data), 1)
        bar_w = (w - 2*pad_x) / n
        sp    = 3

        for i, v in enumerate(data):
            ratio = v / max_v
            bh    = max(4, int(ratio * (h - 2*pad_y)))
            x0 = pad_x + i * bar_w + sp
            x1 = x0 + bar_w - sp*2
            y0 = h - pad_y - bh
            y1 = h - pad_y
            color = GREEN if v >= LOW_BLINK_THRESHOLD else AMBER
            self.create_rectangle(x0, y0+3, x1, y1, fill=color, outline="")
            self.create_oval(x0, y0, x1, y0+6, fill=color, outline="")

        # Eşik çizgisi
        gy = max(pad_y, h - pad_y - int((LOW_BLINK_THRESHOLD/max_v) * (h-2*pad_y)))
        self.create_line(pad_x, gy, w-pad_x, gy, fill=TEXT_DIM, dash=(4, 4))
        self.create_text(w-pad_x+4, gy, text="7", fill=TEXT_DIM,
                         anchor="w", font=("Courier New", 7))


class PulseDot(tk.Canvas):
    """Animasyonlu durum noktası."""

    def __init__(self, parent, **kw):
        super().__init__(parent, width=12, height=12,
                         bg=BG, highlightthickness=0, **kw)
        self._color = TEXT_DIM
        self._tick  = 0
        self._job   = None
        self._draw(1.0)

    def set_color(self, color, animate=False):
        self._color = color
        if self._job:
            self.after_cancel(self._job)
            self._job = None
        if animate:
            self._tick = 0
            self._pulse()
        else:
            self._draw(1.0)

    def _pulse(self):
        self._tick += 1
        alpha = 0.45 + 0.55 * abs(math.sin(self._tick * 0.25))
        self._draw(alpha)
        self._job = self.after(50, self._pulse)

    def _draw(self, alpha):
        self.delete("all")
        r1,g1,b1 = hex_to_rgb(self._color)
        r2,g2,b2 = hex_to_rgb(BG)
        c = "#{:02x}{:02x}{:02x}".format(
            int(r2+(r1-r2)*alpha), int(g2+(g1-g2)*alpha), int(b2+(b1-b2)*alpha))
        self.create_oval(1, 1, 11, 11, fill=c, outline="")


# ── Ana uygulama ───────────────────────────────────────────────────────

class BlinkApp(tk.Tk):

    def __init__(self):
        super().__init__()
        self.title("Blink Counter")
        self.configure(bg=BG)
        self.resizable(False, False)

        # Fontlar
        self._f_huge  = tkfont.Font(family="Courier New", size=72, weight="bold")
        self._f_med   = tkfont.Font(family="Courier New", size=13, weight="bold")
        self._f_small = tkfont.Font(family="Courier New", size=8)
        self._f_label = tkfont.Font(family="Courier New", size=7)

        self._build_ui()
        self._start_detector()
        self.protocol("WM_DELETE_WINDOW", self._on_close)

    # ── Build UI ──────────────────────────────────────────────────────

    def _build_ui(self):
        self._build_header()
        self._sep()
        self._build_count_row()
        self._build_stats_row()
        self._sep()
        self._build_mid_row()
        self._sep()
        self._build_chart_row()
        self._sep()
        self._build_footer()

    def _build_header(self):
        f = tk.Frame(self, bg=BG)
        f.pack(fill="x", padx=20, pady=(16, 8))

        self._dot = PulseDot(f)
        self._dot.pack(side="left")
        tk.Label(f, text=" BLINK COUNTER", bg=BG, fg=TEXT_SEC,
                 font=self._f_small).pack(side="left")

        self._face_lbl = tk.Label(f, text="NO FACE ●", bg=BG,
                                  fg=TEXT_DIM, font=self._f_label)
        self._face_lbl.pack(side="right")

    def _build_count_row(self):
        f = tk.Frame(self, bg=BG)
        f.pack()
        self._count_var = tk.StringVar(value="0")
        tk.Label(f, textvariable=self._count_var,
                 bg=BG, fg=TEXT_PRI, font=self._f_huge).pack()
        tk.Label(f, text="TOTAL BLINKS", bg=BG, fg=TEXT_DIM,
                 font=self._f_label).pack()

    def _build_stats_row(self):
        f = tk.Frame(self, bg=BG)
        f.pack(fill="x", padx=20, pady=(8, 10))
        self._sv_min  = self._stat(f, "LAST MIN")
        self._sv_avg  = self._stat(f, "AVG / MIN")
        self._sv_time = self._stat(f, "ELAPSED")

    def _stat(self, parent, label):
        frame = tk.Frame(parent, bg=PANEL, padx=16, pady=6)
        frame.pack(side="left", expand=True, fill="x", padx=4)
        var = tk.StringVar(value="—")
        tk.Label(frame, textvariable=var, bg=PANEL,
                 fg=TEXT_PRI, font=self._f_med).pack()
        tk.Label(frame, text=label, bg=PANEL,
                 fg=TEXT_SEC, font=self._f_label).pack()
        return var

    def _build_mid_row(self):
        f = tk.Frame(self, bg=BG)
        f.pack(padx=20, pady=10)

        # Kamera
        cam_border = tk.Frame(f, bg=BORDER, padx=1, pady=1)
        cam_border.pack(side="left")
        self._cam_lbl = tk.Label(cam_border, bg=PANEL_LIGHT,
                                 width=320, height=240)
        self._cam_lbl.pack()

        # Sağ panel
        right = tk.Frame(f, bg=BG)
        right.pack(side="left", padx=(14, 0), fill="y")

        tk.Label(right, text="EAR GAUGE", bg=BG,
                 fg=TEXT_SEC, font=self._f_label).pack()
        self._gauge = RingGauge(right, size=122)
        self._gauge.pack(pady=(2, 0))

        # Alarm kutusu
        self._alarm_lbl = tk.Label(
            right, text="BLINKING OK ✓",
            bg=PANEL, fg=GREEN, font=self._f_small,
            padx=10, pady=6)
        self._alarm_lbl.pack(fill="x", pady=(14, 0))

        # EAR bar
        tk.Label(right, text="RAW EAR", bg=BG,
                 fg=TEXT_SEC, font=self._f_label).pack(pady=(12, 0))
        self._ear_canvas = tk.Canvas(right, width=122, height=6,
                                     bg=PANEL_LIGHT, highlightthickness=0)
        self._ear_canvas.pack()

    def _build_chart_row(self):
        f = tk.Frame(self, bg=BG)
        f.pack(fill="x", padx=20, pady=(8, 4))
        tk.Label(f, text="BLINKS PER MINUTE  (last 10 min)",
                 bg=BG, fg=TEXT_SEC, font=self._f_label).pack(anchor="w")
        self._chart = BarChart(f)
        self._chart.pack(pady=(4, 0))

    def _build_footer(self):
        f = tk.Frame(self, bg=BG)
        f.pack(fill="x", padx=20, pady=(8, 16))

        for text, cmd, fg in [
            ("RESET", self._on_reset, TEXT_SEC),
            ("QUIT",  self._on_close, RED_COL),
        ]:
            side = "left" if text == "RESET" else "right"
            tk.Button(
                f, text=text, command=cmd,
                bg=PANEL_LIGHT, fg=fg,
                activebackground=BORDER, activeforeground=TEXT_PRI,
                relief="flat", font=self._f_small,
                padx=16, pady=6, cursor="hand2",
                bd=0
            ).pack(side=side)

    def _sep(self):
        tk.Frame(self, bg=BORDER, height=1).pack(fill="x", padx=20)

    # ── Detector ─────────────────────────────────────────────────────

    def _start_detector(self):
        self._det = BlinkDetector()
        self._det.on_blink      = self._cb_blink
        self._det.on_frame      = self._cb_frame
        self._det.on_alarm      = self._cb_alarm
        self._det.on_face_lost  = lambda: self.after(0, self._set_face, False)
        self._det.on_face_found = lambda: self.after(0, self._set_face, True)
        self._det.start()
        self._tick()

    # ── Callbacks ─────────────────────────────────────────────────────

    def _cb_blink(self, count):
        self.after(0, self._count_var.set, str(count))

    def _cb_frame(self, frame, ear):
        small = cv2.resize(frame, (320, 240))
        rgb   = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
        photo = ImageTk.PhotoImage(Image.fromarray(rgb))
        self.after(0, self._update_cam, photo, ear)

    def _cb_alarm(self):
        self.after(0, self._show_alarm)

    # ── UI updates ────────────────────────────────────────────────────

    def _update_cam(self, photo, ear):
        self._cam_lbl.configure(image=photo)
        self._cam_lbl._photo = photo

        norm = min(ear / 0.35, 1.0)
        self._gauge.set(norm)

        # Raw EAR bar
        self._ear_canvas.delete("all")
        bw = int(norm * 122)
        color = lerp_color(hex_to_rgb(RED_COL), hex_to_rgb(GREEN), norm)
        if bw > 0:
            self._ear_canvas.create_rectangle(0, 0, bw, 6,
                                               fill=color, outline="")

    def _show_alarm(self):
        self._alarm_lbl.configure(text="⚠  BLINK MORE!", fg=AMBER, bg="#221a05")
        if platform.system() == "Windows":
            import winsound
            threading.Thread(target=lambda: winsound.Beep(1000, 800),
                             daemon=True).start()
        self.after(6000, self._clear_alarm)

    def _clear_alarm(self):
        self._alarm_lbl.configure(text="BLINKING OK ✓", fg=GREEN, bg=PANEL)

    def _set_face(self, found: bool):
        if found:
            self._face_lbl.configure(text="FACE FOUND ●", fg=GREEN)
            self._dot.set_color(GREEN, animate=True)
        else:
            self._face_lbl.configure(text="NO FACE ●", fg=TEXT_DIM)
            self._dot.set_color(TEXT_DIM, animate=False)

    # ── Periodic tick ─────────────────────────────────────────────────

    def _tick(self):
        d   = self._det
        sec = d.elapsed_seconds
        self._sv_time.set(f"{int(sec//60)}:{int(sec%60):02d}")
        self._sv_min.set(str(d.blinks_last_minute()))
        if sec > 5:
            self._sv_avg.set(f"{d.blink_count / (sec/60):.1f}")
        self._chart.update_bars(d.minute_history)
        self.after(500, self._tick)

    # ── Controls ──────────────────────────────────────────────────────

    def _on_reset(self):
        self._det.reset()
        self._count_var.set("0")
        self._sv_min.set("—")
        self._sv_avg.set("—")
        self._sv_time.set("0:00")
        self._clear_alarm()

    def _on_close(self):
        self._det.stop()
        self.destroy()


if __name__ == "__main__":
    BlinkApp().mainloop()