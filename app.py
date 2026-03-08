"""
EcoSort — Smart Waste Sorting System
Premium Dark GUI — Pure Tkinter (no ttkbootstrap)
"""

import os
import sys
import math
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk, ImageDraw, ImageFilter, ImageFont

try:
    from inference import load_model, predict
except ImportError:
    print("inference.py not found!")
    sys.exit(1)

# ─── THEME ───────────────────────────────────────────────────────────────────

C = {
    "bg":           "#0a0e1a",
    "panel":        "#111827",
    "card":         "#1a2234",
    "border":       "#1e293b",
    "accent":       "#00d4aa",
    "accent2":      "#6366f1",
    "accent3":      "#f59e0b",
    "white":        "#f1f5f9",
    "gray":         "#64748b",
    "gray2":        "#334155",
    "red":          "#ef4444",
    "green":        "#22c55e",
    "blue":         "#3b82f6",
    "purple":       "#a855f7",
    "orange":       "#f97316",
    "teal":         "#14b8a6",
}

CLASS_COLORS = {
    "cardboard": "#3b82f6",
    "glass":     "#22c55e",
    "metal":     "#f97316",
    "paper":     "#06b6d4",
    "plastic":   "#a855f7",
    "trash":     "#64748b",
}

CLASS_ICONS = {
    "cardboard": "📦",
    "glass":     "🫙",
    "metal":     "🥫",
    "paper":     "📄",
    "plastic":   "🧴",
    "trash":     "🗑️",
}

CKPT_PATH = "ecosort_output/ecosort_best.pth"


# ─── ROUNDED RECTANGLE HELPER ────────────────────────────────────────────────

def rounded_rect(canvas, x1, y1, x2, y2, r=16, **kwargs):
    points = [
        x1+r, y1,  x2-r, y1,
        x2,   y1,  x2,   y1+r,
        x2,   y2-r,x2,   y2,
        x2-r, y2,  x1+r, y2,
        x1,   y2,  x1,   y2-r,
        x1,   y1+r,x1,   y1,
    ]
    return canvas.create_polygon(points, smooth=True, **kwargs)


def make_gradient_image(w, h, color1, color2, vertical=True):
    """Create a gradient PIL image."""
    img = Image.new("RGB", (w, h))
    r1,g1,b1 = int(color1[1:3],16), int(color1[3:5],16), int(color1[5:7],16)
    r2,g2,b2 = int(color2[1:3],16), int(color2[3:5],16), int(color2[5:7],16)
    for i in range(h if vertical else w):
        t = i / (h-1 if vertical else w-1)
        r = int(r1 + (r2-r1)*t)
        g = int(g1 + (g2-g1)*t)
        b = int(b1 + (b2-b1)*t)
        if vertical:
            img.paste((r,g,b), (0,i,w,i+1))
        else:
            img.paste((r,g,b), (i,0,i+1,h))
    return img


# ─── ANIMATED CANVAS BUTTON ──────────────────────────────────────────────────

class FancyButton(tk.Canvas):
    def __init__(self, parent, text, command=None, width=200, height=44,
                 bg_color="#00d4aa", text_color="#0a0e1a", font_size=11,
                 disabled=False, **kwargs):
        super().__init__(parent, width=width, height=height,
                         bg=C["panel"], highlightthickness=0, **kwargs)
        self.command  = command
        self.bg_color = bg_color
        self.text_str = text
        self.w, self.h = width, height
        self.disabled = disabled
        self.hovered  = False

        self._draw(hovered=False)
        self.bind("<Enter>",    self._on_enter)
        self.bind("<Leave>",    self._on_leave)
        self.bind("<Button-1>", self._on_click)

    def _draw(self, hovered=False):
        self.delete("all")
        color = self.bg_color if not self.disabled else C["gray2"]
        if hovered and not self.disabled:
            # Brighten slightly
            r = min(255, int(color[1:3],16)+30)
            g = min(255, int(color[3:5],16)+30)
            b = min(255, int(color[5:7],16)+30)
            color = f"#{r:02x}{g:02x}{b:02x}"

        rounded_rect(self, 2, 2, self.w-2, self.h-2, r=10, fill=color, outline="")
        txt_color = C["white"] if self.disabled else (C["bg"] if not hovered else C["bg"])
        self.create_text(self.w//2, self.h//2, text=self.text_str,
                         font=("Segoe UI", 11, "bold"), fill=txt_color)

    def _on_enter(self, e):
        if not self.disabled:
            self.hovered = True
            self._draw(hovered=True)
            self.config(cursor="hand2")

    def _on_leave(self, e):
        self.hovered = False
        self._draw(hovered=False)

    def _on_click(self, e):
        if not self.disabled and self.command:
            self.command()

    def set_disabled(self, val):
        self.disabled = val
        self._draw(hovered=False)

    def set_text(self, text):
        self.text_str = text
        self._draw(hovered=self.hovered)


# ─── CONFIDENCE BAR ──────────────────────────────────────────────────────────

class ConfBar(tk.Canvas):
    def __init__(self, parent, width=300, height=22, **kwargs):
        super().__init__(parent, width=width, height=height,
                         bg=C["card"], highlightthickness=0, **kwargs)
        self.w, self.h = width, height
        self._pct  = 0
        self._color = C["accent"]
        self._label = ""
        self._draw()

    def set_value(self, pct, color, label=""):
        self._pct   = pct
        self._color = color
        self._label = label
        self._animate_to(pct)

    def _animate_to(self, target, current=0, steps=20):
        if current <= steps:
            val = target * (current / steps)
            self._pct = val
            self._draw()
            self.after(16, lambda: self._animate_to(target, current+1, steps))

    def _draw(self):
        self.delete("all")
        # Background track
        rounded_rect(self, 0, 4, self.w, self.h-4, r=6,
                     fill=C["border"], outline="")
        # Filled bar
        if self._pct > 0:
            fill_w = max(12, int((self.w) * self._pct / 100))
            rounded_rect(self, 0, 4, fill_w, self.h-4, r=6,
                         fill=self._color, outline="")
        # Label
        if self._label:
            self.create_text(8, self.h//2, text=self._label,
                             font=("Segoe UI", 9), fill=C["white"], anchor="w")
        # Percentage
        pct_text = f"{self._pct:.1f}%"
        self.create_text(self.w-4, self.h//2, text=pct_text,
                         font=("Segoe UI", 9, "bold"), fill=C["white"], anchor="e")


# ─── MAIN APP ────────────────────────────────────────────────────────────────

class EcoSortApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("EcoSort — Smart Waste Sorter")
        self.geometry("1160x780")
        self.minsize(900, 640)
        self.configure(bg=C["bg"])

        self.model     = None
        self.pil_image = None
        self.history   = []
        self._spin_angle = 0
        self._spinning   = False

        self._setup_styles()
        self._build_ui()
        self.after(300, self._load_model_async)

    def _setup_styles(self):
        style = ttk.Style()
        style.theme_use("default")
        style.configure("Accent.Horizontal.TProgressbar",
                        troughcolor=C["border"], background=C["accent"],
                        borderwidth=0, lightcolor=C["accent"], darkcolor=C["accent"])

    # ── Model ────────────────────────────────────────────────────────────────

    def _load_model_async(self):
        self._start_spinner()
        def _load():
            try:
                self.model = load_model(CKPT_PATH)
                self.after(0, lambda: [
                    self._stop_spinner(),
                    self._set_status("✅  Model ready  •  94.99% val accuracy", C["green"])
                ])
            except FileNotFoundError:
                self.after(0, lambda: [
                    self._stop_spinner(),
                    self._set_status("⚠️  No model found — run train.py first", C["orange"])
                ])
            except Exception as ex:
                self.after(0, lambda: [
                    self._stop_spinner(),
                    self._set_status(f"❌  {ex}", C["red"])
                ])
        threading.Thread(target=_load, daemon=True).start()

    def _start_spinner(self):
        self._spinning = True
        self._spin()

    def _spin(self):
        if not self._spinning:
            return
        self._spin_angle = (self._spin_angle + 12) % 360
        self._draw_spinner()
        self.after(40, self._spin)

    def _draw_spinner(self):
        if not hasattr(self, "spinner_canvas"):
            return
        c = self.spinner_canvas
        c.delete("all")
        cx, cy, r = 10, 10, 8
        for i in range(8):
            angle = math.radians(self._spin_angle + i * 45)
            x = cx + r * math.cos(angle)
            y = cy + r * math.sin(angle)
            alpha = int(255 * (i+1) / 8)
            color = f"#{alpha//3:02x}{alpha//2:02x}{alpha:02x}"
            c.create_oval(x-2, y-2, x+2, y+2, fill=color, outline="")

    def _stop_spinner(self):
        self._spinning = False
        if hasattr(self, "spinner_canvas"):
            self.spinner_canvas.delete("all")

    # ── UI Build ─────────────────────────────────────────────────────────────

    def _build_ui(self):
        self._build_header()

        content = tk.Frame(self, bg=C["bg"])
        content.pack(fill=tk.BOTH, expand=True, padx=20, pady=(10, 0))

        self._build_left_panel(content)
        self._build_right_panel(content)
        self._build_footer()

    def _build_header(self):
        hdr = tk.Frame(self, bg=C["panel"], height=64)
        hdr.pack(fill=tk.X)
        hdr.pack_propagate(False)

        # Left: logo
        logo_frame = tk.Frame(hdr, bg=C["panel"])
        logo_frame.pack(side=tk.LEFT, padx=24, pady=10)

        logo_badge = tk.Frame(logo_frame, bg=C["accent"], padx=8, pady=2)
        logo_badge.pack(side=tk.LEFT)
        tk.Label(logo_badge, text="🌿", font=("Segoe UI Emoji", 18),
                 bg=C["accent"], fg=C["bg"]).pack(side=tk.LEFT)
        tk.Label(logo_badge, text=" EcoSort", font=("Segoe UI", 17, "bold"),
                 bg=C["accent"], fg=C["bg"]).pack(side=tk.LEFT)

        tk.Label(logo_frame, text="  Smart Waste Classification",
                 font=("Segoe UI", 11), bg=C["panel"], fg=C["gray"]).pack(side=tk.LEFT, padx=10)

        # Right: status
        right_hdr = tk.Frame(hdr, bg=C["panel"])
        right_hdr.pack(side=tk.RIGHT, padx=24)

        self.spinner_canvas = tk.Canvas(right_hdr, width=20, height=20,
                                         bg=C["panel"], highlightthickness=0)
        self.spinner_canvas.pack(side=tk.LEFT, padx=(0, 6))

        self.status_lbl = tk.Label(right_hdr, text="⏳  Initializing…",
                                    font=("Segoe UI", 10), bg=C["panel"], fg=C["gray"])
        self.status_lbl.pack(side=tk.LEFT)

        # Separator line
        tk.Frame(self, bg=C["accent"], height=2).pack(fill=tk.X)

    def _build_left_panel(self, parent):
        left = tk.Frame(parent, bg=C["panel"], width=400)
        left.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 12), pady=(0, 12))
        left.pack_propagate(False)

        # Section title
        self._section_title(left, "INPUT IMAGE")

        # Image drop zone
        drop_frame = tk.Frame(left, bg=C["border"], padx=2, pady=2)
        drop_frame.pack(padx=16, pady=(0, 14))

        self.img_canvas = tk.Canvas(drop_frame, width=356, height=320,
                                     bg=C["card"], highlightthickness=0)
        self.img_canvas.pack()
        self._draw_placeholder()
        self.img_canvas.bind("<Button-1>", lambda e: self._open_image())
        self.img_canvas.bind("<Enter>",    self._on_canvas_enter)
        self.img_canvas.bind("<Leave>",    self._on_canvas_leave)

        # Buttons
        btn_row = tk.Frame(left, bg=C["panel"])
        btn_row.pack(fill=tk.X, padx=16, pady=(0, 10))

        self.open_btn = FancyButton(btn_row, text="📂  Open Image",
                                     command=self._open_image, width=176, height=40,
                                     bg_color=C["accent2"])
        self.open_btn.pack(side=tk.LEFT)

        tk.Frame(btn_row, bg=C["panel"], width=8).pack(side=tk.LEFT)

        self.cam_btn = FancyButton(btn_row, text="📷  Camera",
                                    command=self._capture_camera, width=176, height=40,
                                    bg_color=C["blue"])
        self.cam_btn.pack(side=tk.LEFT)

        # Classify button
        self.classify_btn = FancyButton(
            left, text="🔍  CLASSIFY WASTE",
            command=self._classify, width=356, height=52,
            bg_color=C["accent"], disabled=True
        )
        self.classify_btn.pack(padx=16, pady=(0, 10))

        # Inference progress
        self.inf_progress = ttk.Progressbar(left, style="Accent.Horizontal.TProgressbar",
                                             mode="indeterminate", length=356)
        self.inf_progress.pack(padx=16)

        # File info
        self.file_lbl = tk.Label(left, text="No image loaded",
                                  font=("Segoe UI", 9), bg=C["panel"], fg=C["gray"])
        self.file_lbl.pack(pady=(6, 0))

    def _build_right_panel(self, parent):
        right = tk.Frame(parent, bg=C["bg"])
        right.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, pady=(0, 12))

        # Top row: result card + stats
        top_row = tk.Frame(right, bg=C["bg"])
        top_row.pack(fill=tk.X, pady=(0, 12))

        self._build_result_card(top_row)
        self._build_info_card(top_row)

        # Top-3 predictions
        self._section_title(right, "TOP-3 PREDICTIONS")
        self.topk_frame = tk.Frame(right, bg=C["panel"])
        self.topk_frame.pack(fill=tk.X, padx=0, pady=(0, 12))
        self._build_topk_bars()

        # Recycling tip card
        self._section_title(right, "RECYCLING GUIDE")
        self.tip_card = tk.Frame(right, bg=C["card"])
        self.tip_card.pack(fill=tk.X, pady=(0, 12))

        tip_inner = tk.Frame(self.tip_card, bg=C["card"])
        tip_inner.pack(fill=tk.X, padx=16, pady=12)

        self.tip_icon = tk.Label(tip_inner, text="♻️", font=("Segoe UI Emoji", 28),
                                  bg=C["card"], fg=C["white"])
        self.tip_icon.pack(side=tk.LEFT, padx=(0, 12))

        tip_text_frame = tk.Frame(tip_inner, bg=C["card"])
        tip_text_frame.pack(side=tk.LEFT, fill=tk.X, expand=True)

        self.tip_title = tk.Label(tip_text_frame, text="Open an image to get started",
                                   font=("Segoe UI", 11, "bold"), bg=C["card"], fg=C["white"],
                                   anchor=tk.W)
        self.tip_title.pack(anchor=tk.W)

        self.tip_body = tk.Label(tip_text_frame,
                                  text="Upload a photo of your waste item and click Classify.",
                                  font=("Segoe UI", 10), bg=C["card"], fg=C["gray"],
                                  anchor=tk.W, wraplength=460, justify=tk.LEFT)
        self.tip_body.pack(anchor=tk.W, pady=(2, 0))

    def _build_result_card(self, parent):
        card = tk.Frame(parent, bg=C["panel"], width=240)
        card.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 12))
        card.pack_propagate(False)

        self._section_title(card, "RESULT")

        self.result_canvas = tk.Canvas(card, width=220, height=180,
                                        bg=C["panel"], highlightthickness=0)
        self.result_canvas.pack(padx=10, pady=(0, 8))
        self._draw_result_placeholder()

        self.conf_bar_lbl = tk.Label(card, text="Confidence", font=("Segoe UI", 9),
                                      bg=C["panel"], fg=C["gray"])
        self.conf_bar_lbl.pack(anchor=tk.W, padx=12)

        self.conf_bar = ConfBar(card, width=216, height=24)
        self.conf_bar.pack(padx=12, pady=(2, 10))

    def _build_info_card(self, parent):
        card = tk.Frame(parent, bg=C["panel"])
        card.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self._section_title(card, "WASTE INFO")

        info_inner = tk.Frame(card, bg=C["panel"])
        info_inner.pack(fill=tk.BOTH, expand=True, padx=14, pady=(0, 14))

        def info_row(label, default="—"):
            row = tk.Frame(info_inner, bg=C["card"])
            row.pack(fill=tk.X, pady=3)
            tk.Label(row, text=label, font=("Segoe UI", 9, "bold"),
                     bg=C["card"], fg=C["gray"], width=12, anchor=tk.W).pack(side=tk.LEFT, padx=(10,4), pady=6)
            val = tk.Label(row, text=default, font=("Segoe UI", 10),
                           bg=C["card"], fg=C["white"], anchor=tk.W, wraplength=280)
            val.pack(side=tk.LEFT, padx=(0,10), pady=6)
            return val

        self.info_class   = info_row("Category")
        self.info_bin     = info_row("Bin Color")
        self.info_recycle = info_row("Recyclable")
        self.info_co2     = info_row("CO₂ Saved")

        # Badge
        self.badge_canvas = tk.Canvas(info_inner, width=200, height=36,
                                       bg=C["panel"], highlightthickness=0)
        self.badge_canvas.pack(anchor=tk.W, pady=(8, 0))
        self._draw_badge("—", C["gray2"])

    def _build_topk_bars(self):
        self.bar_rows = []
        for i in range(3):
            row = tk.Frame(self.topk_frame, bg=C["panel"])
            row.pack(fill=tk.X, padx=14, pady=4)

            rank = tk.Label(row, text=f"#{i+1}", font=("Segoe UI", 10, "bold"),
                            bg=C["panel"], fg=C["gray"], width=3)
            rank.pack(side=tk.LEFT)

            lbl = tk.Label(row, text="—", font=("Segoe UI", 10),
                           bg=C["panel"], fg=C["white"], width=12, anchor=tk.W)
            lbl.pack(side=tk.LEFT, padx=(4, 8))

            bar = ConfBar(row, width=260, height=24)
            bar.pack(side=tk.LEFT, fill=tk.X, expand=True)

            self.bar_rows.append((lbl, bar))

        tk.Frame(self.topk_frame, bg=C["panel"], height=6).pack()

    def _section_title(self, parent, text):
        row = tk.Frame(parent, bg=parent.cget("bg"))
        row.pack(fill=tk.X, padx=14, pady=(10, 6))
        tk.Frame(row, bg=C["accent"], width=3, height=16).pack(side=tk.LEFT)
        tk.Label(row, text=f"  {text}", font=("Segoe UI", 9, "bold"),
                 bg=parent.cget("bg"), fg=C["gray"]).pack(side=tk.LEFT)

    def _build_footer(self):
        footer = tk.Frame(self, bg=C["panel"], height=44)
        footer.pack(fill=tk.X, side=tk.BOTTOM)
        footer.pack_propagate(False)

        tk.Frame(self, bg=C["border"], height=1).pack(fill=tk.X, side=tk.BOTTOM)

        tk.Label(footer, text="History:", font=("Segoe UI", 9, "bold"),
                 bg=C["panel"], fg=C["gray"]).pack(side=tk.LEFT, padx=16, pady=10)

        self.history_frame = tk.Frame(footer, bg=C["panel"])
        self.history_frame.pack(side=tk.LEFT, pady=8)

        tk.Label(footer, text="EcoSort v1.0  •  EfficientNetV2-M  •  RTX 3050",
                 font=("Segoe UI", 9), bg=C["panel"], fg=C["gray2"]).pack(side=tk.RIGHT, padx=16)

    # ── Canvas Drawing ────────────────────────────────────────────────────────

    def _draw_placeholder(self):
        self.img_canvas.delete("all")
        c = self.img_canvas
        # Dashed border
        for i in range(0, 356, 16):
            c.create_line(i, 0, i+8, 0, fill=C["gray2"], width=1)
            c.create_line(i, 320, i+8, 320, fill=C["gray2"], width=1)
        for i in range(0, 320, 16):
            c.create_line(0, i, 0, i+8, fill=C["gray2"], width=1)
            c.create_line(356, i, 356, i+8, fill=C["gray2"], width=1)
        c.create_text(178, 130, text="🖼️", font=("Segoe UI Emoji", 44), fill=C["gray2"])
        c.create_text(178, 192, text="Click to open an image",
                      font=("Segoe UI", 12), fill=C["gray"])
        c.create_text(178, 216, text="JPG  •  PNG  •  BMP  •  WEBP",
                      font=("Segoe UI", 9), fill=C["gray2"])

    def _draw_result_placeholder(self):
        c = self.result_canvas
        c.delete("all")
        cx, cy = 110, 90
        c.create_oval(cx-50, cy-50, cx+50, cy+50, outline=C["gray2"], width=2, dash=(4,4))
        c.create_text(cx, cy, text="?", font=("Segoe UI", 36, "bold"), fill=C["gray2"])
        c.create_text(cx, 152, text="No classification yet",
                      font=("Segoe UI", 9), fill=C["gray"])

    def _draw_result(self, label, conf, color):
        c = self.result_canvas
        c.delete("all")
        cx, cy = 110, 80

        # Glow circle
        for i in range(5, 0, -1):
            alpha = int(40 * i / 5)
            r = 52 + (5-i)*4
            hex_a = f"{alpha:02x}"
            # Simulate glow with multiple circles
            c.create_oval(cx-r, cy-r, cx+r, cy+r,
                          outline=color, width=1)

        # Main circle
        c.create_oval(cx-48, cy-48, cx+48, cy+48, fill=color, outline="")

        # Icon
        icon = CLASS_ICONS.get(label, "?")
        c.create_text(cx, cy, text=icon, font=("Segoe UI Emoji", 30))

        # Label
        c.create_text(cx, cy+70, text=label.upper(),
                      font=("Segoe UI", 13, "bold"), fill=color)

        # Confidence text
        c.create_text(cx, cy+92, text=f"{conf*100:.1f}% confidence",
                      font=("Segoe UI", 9), fill=C["gray"])

    def _draw_badge(self, text, color):
        c = self.badge_canvas
        c.delete("all")
        if text == "—":
            return
        rounded_rect(c, 0, 4, 190, 32, r=8, fill=color, outline="")
        c.create_text(95, 18, text=text, font=("Segoe UI", 10, "bold"),
                      fill=C["white"])

    def _on_canvas_enter(self, e):
        self.img_canvas.config(bg="#1e2a3a")

    def _on_canvas_leave(self, e):
        self.img_canvas.config(bg=C["card"])

    # ── Image Handling ────────────────────────────────────────────────────────

    def _open_image(self):
        path = filedialog.askopenfilename(
            title="Select Waste Image",
            filetypes=[("Images", "*.jpg *.jpeg *.png *.bmp *.webp *.tiff")]
        )
        if path:
            self.pil_image = Image.open(path).convert("RGB")
            self._show_image(self.pil_image)
            self.classify_btn.set_disabled(False)
            fname = os.path.basename(path)
            size  = os.path.getsize(path) // 1024
            self.file_lbl.config(text=f"📄  {fname}  ({size} KB)", fg=C["accent"])
            self._set_status(f"Image loaded: {fname}", C["blue"])

    def _show_image(self, pil_img):
        disp = pil_img.copy()
        disp.thumbnail((356, 320))
        tk_img = ImageTk.PhotoImage(disp)
        self.img_canvas.delete("all")
        x = (356 - disp.width)  // 2
        y = (320 - disp.height) // 2
        self.img_canvas.create_image(x, y, anchor=tk.NW, image=tk_img)
        self.img_canvas._photo = tk_img

    def _capture_camera(self):
        try:
            import cv2
        except ImportError:
            messagebox.showinfo("OpenCV needed", "Run: pip install opencv-python")
            return

        # Open live camera preview window
        cam_win = tk.Toplevel(self)
        cam_win.title("📷  Camera Preview")
        cam_win.configure(bg=C["bg"])
        cam_win.resizable(False, False)
        cam_win.grab_set()  # Modal

        # Header
        tk.Label(cam_win, text="📷  Live Camera Preview",
                 font=("Segoe UI", 12, "bold"),
                 bg=C["panel"], fg=C["white"]).pack(fill=tk.X, padx=0, pady=0, ipadx=16, ipady=10)

        tk.Frame(cam_win, bg=C["accent"], height=2).pack(fill=tk.X)

        # Live feed canvas
        feed_canvas = tk.Canvas(cam_win, width=640, height=440,
                                bg=C["card"], highlightthickness=0)
        feed_canvas.pack(padx=16, pady=16)

        tk.Label(cam_win, text="Position your waste item in the frame",
                 font=("Segoe UI", 10), bg=C["bg"], fg=C["gray"]).pack()

        # Buttons row
        btn_row = tk.Frame(cam_win, bg=C["bg"])
        btn_row.pack(pady=12)

        captured_frame = [None]  # mutable container

        def do_capture():
            if captured_frame[0] is not None:
                rgb = cv2.cvtColor(captured_frame[0], cv2.COLOR_BGR2RGB)
                self.pil_image = Image.fromarray(rgb)
                stop_feed()
                cam_win.destroy()
                self._show_image(self.pil_image)
                self.classify_btn.set_disabled(False)
                self.file_lbl.config(text="📷  Webcam capture", fg=C["accent"])
                self._set_status("Camera capture loaded — click Classify!", C["blue"])

        def do_cancel():
            stop_feed()
            cam_win.destroy()

        cap_btn = FancyButton(btn_row, text="📸  Capture",
                               command=do_capture, width=180, height=44,
                               bg_color=C["accent"])
        cap_btn.pack(side=tk.LEFT, padx=6)

        FancyButton(btn_row, text="✕  Cancel",
                     command=do_cancel, width=120, height=44,
                     bg_color=C["gray2"]).pack(side=tk.LEFT, padx=6)

        # Flash effect label
        flash_lbl = tk.Label(cam_win, text="", font=("Segoe UI", 10, "bold"),
                              bg=C["bg"], fg=C["green"])
        flash_lbl.pack(pady=(0, 8))

        # Start live feed in background thread
        running = [True]
        cap = cv2.VideoCapture(0)

        if not cap.isOpened():
            messagebox.showwarning("Camera", "Could not open webcam.")
            cam_win.destroy()
            return

        # Set resolution
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 440)
        cap.set(cv2.CAP_PROP_FPS, 30)

        def feed_loop():
            while running[0]:
                ret, frame = cap.read()
                if not ret:
                    break
                captured_frame[0] = frame.copy()
                rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img   = Image.fromarray(rgb)
                img.thumbnail((640, 440))
                tk_img = ImageTk.PhotoImage(img)
                try:
                    feed_canvas.delete("all")
                    feed_canvas.create_image(0, 0, anchor=tk.NW, image=tk_img)
                    feed_canvas._photo = tk_img
                except tk.TclError:
                    break

        def stop_feed():
            running[0] = False
            cap.release()

        cam_win.protocol("WM_DELETE_WINDOW", do_cancel)

        # Run feed in thread
        threading.Thread(target=feed_loop, daemon=True).start()

    # ── Classification ────────────────────────────────────────────────────────

    def _classify(self):
        if not self.model:
            messagebox.showerror("Not Ready", "Model not loaded yet.")
            return
        if not self.pil_image:
            return
        self.classify_btn.set_disabled(True)
        self.inf_progress.start(8)
        self._set_status("🔍  Classifying…", C["accent"])
        threading.Thread(target=self._run_inference, daemon=True).start()

    def _run_inference(self):
        try:
            result = predict(self.model, self.pil_image)
            self.after(0, lambda: self._show_result(result))
        except Exception as e:
            self.after(0, lambda: messagebox.showerror("Error", str(e)))
        finally:
            self.after(0, self._stop_inference)

    def _stop_inference(self):
        self.inf_progress.stop()
        self.classify_btn.set_disabled(False)

    def _show_result(self, result):
        label = result["label"]
        conf  = result["confidence"]
        info  = result["info"]
        topk  = result["topk"]
        color = CLASS_COLORS.get(label, C["gray"])

        # Result card
        self._draw_result(label, conf, color)
        self.conf_bar.set_value(conf*100, color)

        # Info card
        self.info_class.config(text=f"{CLASS_ICONS.get(label,'')}  {label.capitalize()}", fg=color)
        self.info_bin.config(text=info["bin_color"])
        rec_text = "✅  Yes" if info["recyclable"] else "❌  No"
        rec_col  = C["green"] if info["recyclable"] else C["red"]
        self.info_recycle.config(text=rec_text, fg=rec_col)
        self.info_co2.config(text=info["co2_saved"], fg=C["accent"])

        badge_text  = "♻️  RECYCLABLE" if info["recyclable"] else "🗑️  GENERAL WASTE"
        badge_color = C["green"] if info["recyclable"] else C["gray"]
        self._draw_badge(badge_text, badge_color)

        # Top-3 bars
        for i, (lbl_w, bar_w) in enumerate(self.bar_rows):
            if i < len(topk):
                l, p = topk[i]
                c = CLASS_COLORS.get(l, C["gray"])
                lbl_w.config(text=CLASS_ICONS.get(l,"")+" "+l, fg=C["white"] if l==label else C["gray"])
                bar_w.set_value(p*100, c, "")
            else:
                lbl_w.config(text="—", fg=C["gray"])

        # Tip card
        self.tip_icon.config(text=info["icon"])
        self.tip_title.config(text=f"Tip for {label.capitalize()}", fg=color)
        self.tip_body.config(text=info["tip"])

        # History chip
        self._add_history(label, color)
        self._set_status(
            f"✅  Classified as {label.upper()}  •  {conf*100:.1f}% confidence", C["green"]
        )

    def _add_history(self, label, color):
        self.history.append(label)
        for w in self.history_frame.winfo_children():
            w.destroy()
        for lbl in self.history[-10:]:
            c = CLASS_COLORS.get(lbl, C["gray"])
            chip = tk.Label(
                self.history_frame,
                text=f"{CLASS_ICONS.get(lbl,'')} {lbl[:4]}",
                font=("Segoe UI", 9, "bold"),
                bg=c, fg="white", padx=7, pady=2
            )
            chip.pack(side=tk.LEFT, padx=2)

    # ── Helpers ──────────────────────────────────────────────────────────────

    def _set_status(self, msg, color=None):
        self.status_lbl.config(text=msg, fg=color or C["gray"])


def main():
    app = EcoSortApp()
    app.mainloop()


if __name__ == "__main__":
    main()