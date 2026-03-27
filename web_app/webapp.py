"""
ASL Interpreter — Local Desktop Application
=============================================
A Tkinter GUI that uses your trained Keras CNN + MediaPipe Hand Landmarker
to recognise ASL signs from your webcam in real time.

Layout:
    ┌──────────────────────┬───────────────────────────┐
    │  Camera Feed         │  Detected Letter          │
    │  (top-left)          │  + hold progress bar      │
    │                      │                           │
    ├──────────────────────┤  Sentence Builder         │
    │  Hand Crop           │  (with cursor)            │
    │  (bottom-left)       │                           │
    │                      │  [Delete]  [Clear]        │
    │  [▶ Start] [■ Stop]  │                           │
    └──────────────────────┴───────────────────────────┘

Requirements:
    pip install tensorflow opencv-python mediapipe numpy pillow

Usage:
    python asl_app.py

    Override the model path with an environment variable if needed:
        set ASL_MODEL_PATH=C:\\path\\to\\model.keras
        python asl_app.py
"""

import os
import time
import threading
from collections import deque

import cv2
import numpy as np
import tensorflow as tf
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

from PIL import Image, ImageTk

import tkinter as tk
from tkinter import font as tkfont

# =============================================================================
# CONFIG
# =============================================================================
MODEL_PATH = os.environ.get(
    "ASL_MODEL_PATH",
    r"C:\Users\jvond\ML_Project\asl_finetuned\asl_model_finetuned.keras",
)
HAND_LANDMARKER_PATH = os.environ.get(
    "ASL_HAND_LANDMARKER",
    r"C:\Users\jvond\ML_Project\hand_landmarker.task",
)

IMG_SIZE = 96
CAMERA_INDEX = 0
CONFIDENCE_THRESHOLD = 0.70
PADDING = 80
NORMALIZE_INPUT = False  # Model has built-in Rescaling(1./255)
PREDICTION_HISTORY = 5
LETTER_HOLD_SECONDS = 3

CLASS_NAMES = [
    "A", "B", "C", "D", "E", "F", "G", "H", "I", "J",
    "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T",
    "U", "V", "W", "X", "Y", "Z", "del", "space", "nothing",
]

# =============================================================================
# COLOUR PALETTE
# =============================================================================
BG_DARK      = "#0B0E11"
BG_PANEL     = "#12161C"
BG_CARD      = "#1A1F28"
ACCENT_GREEN = "#00E676"
ACCENT_RED   = "#FF3D71"
ACCENT_AMBER = "#FFAB00"
TEXT_PRIMARY  = "#E8ECF0"
TEXT_DIM      = "#5A6270"
BORDER        = "#2A3040"


# =============================================================================
# HELPERS (same logic as your original pipeline)
# =============================================================================
def make_square_bbox(x_min, y_min, x_max, y_max, fw, fh):
    """Expand bbox to a square, shifting rather than shrinking."""
    side = max(x_max - x_min, y_max - y_min)
    cx = (x_min + x_max) // 2
    cy = (y_min + y_max) // 2
    half = side // 2

    nx0, ny0 = cx - half, cy - half
    nx1, ny1 = cx + half, cy + half

    if nx0 < 0:
        nx1 -= nx0; nx0 = 0
    if ny0 < 0:
        ny1 -= ny0; ny0 = 0
    if nx1 > fw:
        nx0 -= (nx1 - fw); nx1 = fw
    if ny1 > fh:
        ny0 -= (ny1 - fh); ny1 = fh

    return max(nx0, 0), max(ny0, 0), min(nx1, fw), min(ny1, fh)


def get_hand_bbox(hand_landmarks, fw, fh, padding=80):
    """Convert MediaPipe normalised landmarks into a padded pixel bbox."""
    xs = [lm.x for lm in hand_landmarks]
    ys = [lm.y for lm in hand_landmarks]
    x0 = max(int(min(xs) * fw) - padding, 0)
    y0 = max(int(min(ys) * fh) - padding, 0)
    x1 = min(int(max(xs) * fw) + padding, fw)
    y1 = min(int(max(ys) * fh) + padding, fh)
    return x0, y0, x1, y1


def preprocess_crop(crop_bgr):
    """Resize, recolour, and batch a hand crop for the CNN."""
    resized = cv2.resize(crop_bgr, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    arr = rgb.astype(np.float32)
    if NORMALIZE_INPUT:
        arr /= 255.0
    return np.expand_dims(arr, axis=0)


# =============================================================================
# APPLICATION
# =============================================================================
class ASLApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("ASL Interpreter")
        self.root.configure(bg=BG_DARK)
        self.root.minsize(1100, 700)

        # State
        self.running = False
        self.cap = None
        self.model = None
        self.landmarker = None
        self.thread = None

        self.recent_preds: deque = deque(maxlen=PREDICTION_HISTORY)
        self.hold_letter = None
        self.hold_start = 0.0
        self.last_confirmed_key = None
        self.sentence = ""
        self.current_letter = None
        self.current_conf = 0.0
        self.hold_progress = 0.0
        self.hand_detected = False

        # Fonts
        self.font_title    = tkfont.Font(family="Segoe UI", size=13, weight="bold")
        self.font_label    = tkfont.Font(family="Segoe UI", size=10)
        self.font_small    = tkfont.Font(family="Consolas", size=9)
        self.font_letter   = tkfont.Font(family="Segoe UI", size=52, weight="bold")
        self.font_sentence = tkfont.Font(family="Segoe UI", size=22)
        self.font_button   = tkfont.Font(family="Segoe UI", size=11, weight="bold")
        self.font_status   = tkfont.Font(family="Consolas", size=9)

        self._build_ui()
        self._load_models()

    # ─────────────────────────────────────────────
    # BUILD THE INTERFACE
    # ─────────────────────────────────────────────
    def _build_ui(self):
        # ── Header bar ──
        header = tk.Frame(self.root, bg=BG_PANEL, height=50)
        header.pack(fill="x", side="top")
        header.pack_propagate(False)

        tk.Label(
            header, text="✋  ASL INTERPRETER", font=self.font_title,
            fg=ACCENT_GREEN, bg=BG_PANEL,
        ).pack(side="left", padx=16)

        self.status_dot = tk.Label(header, text="●", font=self.font_label,
                                   fg=ACCENT_RED, bg=BG_PANEL)
        self.status_dot.pack(side="right", padx=(0, 6))
        self.status_label = tk.Label(header, text="OFFLINE", font=self.font_small,
                                     fg=ACCENT_RED, bg=BG_PANEL)
        self.status_label.pack(side="right", padx=(0, 2))

        # ── Main two-column grid ──
        main = tk.Frame(self.root, bg=BG_DARK)
        main.pack(fill="both", expand=True, padx=12, pady=12)
        main.columnconfigure(0, weight=3, minsize=480)
        main.columnconfigure(1, weight=2, minsize=350)
        main.rowconfigure(0, weight=1)

        # ════════════════════════════════════════
        # LEFT COLUMN
        # ════════════════════════════════════════
        left = tk.Frame(main, bg=BG_DARK)
        left.grid(row=0, column=0, sticky="nsew", padx=(0, 8))
        left.rowconfigure(0, weight=3)
        left.rowconfigure(1, weight=0)
        left.columnconfigure(0, weight=1)

        # Camera feed
        cam_frame = tk.Frame(left, bg=BG_CARD,
                             highlightbackground=BORDER, highlightthickness=1)
        cam_frame.grid(row=0, column=0, sticky="nsew", pady=(0, 8))

        tk.Label(cam_frame, text="CAMERA FEED", font=self.font_small,
                 fg=TEXT_DIM, bg=BG_CARD, anchor="w").pack(fill="x", padx=10, pady=(8, 0))

        self.cam_label = tk.Label(cam_frame, bg="#000000")
        self.cam_label.pack(fill="both", expand=True, padx=6, pady=6)

        # Bottom-left: crop + controls
        bot_left = tk.Frame(left, bg=BG_DARK)
        bot_left.grid(row=1, column=0, sticky="ew")
        bot_left.columnconfigure(1, weight=1)

        # Hand crop
        crop_card = tk.Frame(bot_left, bg=BG_CARD,
                             highlightbackground=BORDER, highlightthickness=1,
                             width=200, height=230)
        crop_card.grid(row=0, column=0, sticky="ns", padx=(0, 8))
        crop_card.pack_propagate(False)

        tk.Label(crop_card, text="HAND CROP", font=self.font_small,
                 fg=TEXT_DIM, bg=BG_CARD, anchor="w").pack(fill="x", padx=8, pady=(6, 0))

        self.crop_label = tk.Label(crop_card, bg="#000000", width=180, height=180)
        self.crop_label.pack(padx=8, pady=8)

        # Controls
        ctrl_card = tk.Frame(bot_left, bg=BG_CARD,
                             highlightbackground=BORDER, highlightthickness=1)
        ctrl_card.grid(row=0, column=1, sticky="nsew")

        btn_row = tk.Frame(ctrl_card, bg=BG_CARD)
        btn_row.pack(fill="x", padx=12, pady=(16, 8))

        self.start_btn = tk.Button(
            btn_row, text="▶  START", font=self.font_button,
            bg=ACCENT_GREEN, fg=BG_DARK, activebackground="#00C864",
            activeforeground=BG_DARK, relief="flat", cursor="hand2",
            padx=20, pady=8, command=self._start,
        )
        self.start_btn.pack(side="left", padx=(0, 8))

        self.stop_btn = tk.Button(
            btn_row, text="■  STOP", font=self.font_button,
            bg=ACCENT_RED, fg="#FFFFFF", activebackground="#D6305A",
            activeforeground="#FFFFFF", relief="flat", cursor="hand2",
            padx=20, pady=8, command=self._stop, state="disabled",
        )
        self.stop_btn.pack(side="left")

        self.info_label = tk.Label(ctrl_card, text="Press START to begin",
                                   font=self.font_status, fg=TEXT_DIM,
                                   bg=BG_CARD, anchor="w", justify="left")
        self.info_label.pack(fill="x", padx=14, pady=(4, 4))

        self.conf_label = tk.Label(ctrl_card, text="Confidence: —",
                                   font=self.font_status, fg=TEXT_DIM,
                                   bg=BG_CARD, anchor="w")
        self.conf_label.pack(fill="x", padx=14, pady=(0, 4))

        self.threshold_label = tk.Label(
            ctrl_card,
            text=f"Threshold: {CONFIDENCE_THRESHOLD:.0%}  •  Hold: {LETTER_HOLD_SECONDS}s",
            font=self.font_status, fg=TEXT_DIM, bg=BG_CARD, anchor="w",
        )
        self.threshold_label.pack(fill="x", padx=14, pady=(0, 12))

        # ════════════════════════════════════════
        # RIGHT COLUMN
        # ════════════════════════════════════════
        right = tk.Frame(main, bg=BG_DARK)
        right.grid(row=0, column=1, sticky="nsew", padx=(8, 0))
        right.rowconfigure(1, weight=1)
        right.columnconfigure(0, weight=1)

        # Detected letter card
        letter_card = tk.Frame(right, bg=BG_CARD,
                               highlightbackground=BORDER, highlightthickness=1)
        letter_card.grid(row=0, column=0, sticky="ew", pady=(0, 8))

        tk.Label(letter_card, text="DETECTED SIGN", font=self.font_small,
                 fg=TEXT_DIM, bg=BG_CARD).pack(pady=(12, 0))

        self.letter_label = tk.Label(letter_card, text="—", font=self.font_letter,
                                     fg=TEXT_DIM, bg=BG_CARD)
        self.letter_label.pack(pady=(4, 4))

        # Progress bar
        self.progress_frame = tk.Frame(letter_card, bg=BG_DARK, height=8)
        self.progress_frame.pack(fill="x", padx=20, pady=(0, 4))
        self.progress_frame.pack_propagate(False)

        self.progress_bar = tk.Frame(self.progress_frame, bg=ACCENT_GREEN, height=8)
        self.progress_bar.place(x=0, y=0, relheight=1.0, width=0)

        self.hold_label = tk.Label(
            letter_card, text="Hold a sign for 3 seconds to confirm",
            font=self.font_small, fg=TEXT_DIM, bg=BG_CARD,
        )
        self.hold_label.pack(pady=(0, 12))

        # Sentence builder card
        sent_card = tk.Frame(right, bg=BG_CARD,
                             highlightbackground=BORDER, highlightthickness=1)
        sent_card.grid(row=1, column=0, sticky="nsew", pady=(0, 8))
        sent_card.rowconfigure(1, weight=1)
        sent_card.columnconfigure(0, weight=1)

        sent_header = tk.Frame(sent_card, bg=BG_CARD)
        sent_header.grid(row=0, column=0, sticky="ew", padx=12, pady=(10, 0))

        tk.Label(sent_header, text="SENTENCE BUILDER", font=self.font_small,
                 fg=TEXT_DIM, bg=BG_CARD).pack(side="left")

        tk.Button(
            sent_header, text="CLEAR", font=self.font_small,
            bg=BG_PANEL, fg=TEXT_DIM, relief="flat", cursor="hand2",
            activebackground=BG_DARK, activeforeground=TEXT_PRIMARY,
            padx=10, pady=2, command=self._clear_sentence,
        ).pack(side="right", padx=(4, 0))

        tk.Button(
            sent_header, text="← DELETE", font=self.font_small,
            bg=BG_PANEL, fg=TEXT_DIM, relief="flat", cursor="hand2",
            activebackground=BG_DARK, activeforeground=TEXT_PRIMARY,
            padx=10, pady=2, command=self._delete_last,
        ).pack(side="right")

        self.sentence_label = tk.Label(
            sent_card, text="", font=self.font_sentence,
            fg=TEXT_PRIMARY, bg=BG_CARD, anchor="nw",
            justify="left", wraplength=340,
        )
        self.sentence_label.grid(row=1, column=0, sticky="nsew", padx=16, pady=12)

        # ASL Reference Chart — grid of hand signal images with letter labels
        alpha_card = tk.Frame(right, bg=BG_CARD,
                              highlightbackground=BORDER, highlightthickness=1)
        alpha_card.grid(row=2, column=0, sticky="nsew")
        right.rowconfigure(2, weight=2)  # give reference chart space

        tk.Label(alpha_card, text="ASL ALPHABET REFERENCE", font=self.font_small,
                 fg=TEXT_DIM, bg=BG_CARD).pack(pady=(6, 2))

        ref_inner = tk.Frame(alpha_card, bg=BG_CARD)
        ref_inner.pack(fill="both", expand=True, padx=4, pady=(0, 4))

        self._ref_photos = {}  # prevent GC for all loaded images
        self.alpha_labels = {}
        self._build_image_reference(ref_inner)

    # ─────────────────────────────────────────────
    # ASL IMAGE REFERENCE GRID
    # ─────────────────────────────────────────────
    def _build_image_reference(self, parent):
        """Build a grid of hand signal images (A.jpg–Z.jpg) with letter labels."""
        # Path to the folder containing A.jpg, B.jpg, … Z.jpg
        SIGNS_DIR = r"C:\Users\jvond\ML_Project\web_app\input_data"

        COLS = 7  # 7 columns × 4 rows
        THUMB_SIZE = 48  # thumbnail size in pixels

        for i, ch in enumerate("ABCDEFGHIJKLMNOPQRSTUVWXYZ"):
            r, c = divmod(i, COLS)
            parent.columnconfigure(c, weight=1)
            parent.rowconfigure(r, weight=1)

            cell = tk.Frame(parent, bg=BG_PANEL,
                            highlightbackground=BORDER, highlightthickness=1)
            cell.grid(row=r, column=c, padx=1, pady=1, sticky="nsew")

            # Try to load the hand signal image (.jpg first, then .png)
            img_label = None
            for ext in [".jpg", ".png", ".jpeg"]:
                img_path = os.path.join(SIGNS_DIR, f"{ch}{ext}")
                if os.path.exists(img_path):
                    try:
                        img = Image.open(img_path)
                        img = img.resize((THUMB_SIZE, THUMB_SIZE), Image.LANCZOS)
                        photo = ImageTk.PhotoImage(img)
                        self._ref_photos[ch] = photo  # prevent GC
                        img_label = tk.Label(cell, image=photo, bg=BG_PANEL)
                        img_label.pack(padx=2, pady=(3, 0))
                    except Exception:
                        pass
                    break

            if img_label is None:
                placeholder = tk.Label(cell, text="?", font=("Consolas", 16),
                                       fg=TEXT_DIM, bg=BG_PANEL, width=4, height=2)
                placeholder.pack(padx=2, pady=(3, 0))

            # Letter label underneath
            letter_lbl = tk.Label(cell, text=ch, font=("Consolas", 10, "bold"),
                                  fg=TEXT_DIM, bg=BG_PANEL, anchor="center")
            letter_lbl.pack(pady=(0, 3))

            self.alpha_labels[ch] = (cell, letter_lbl)

    # ─────────────────────────────────────────────
    # LOAD MODELS
    # ─────────────────────────────────────────────
    def _load_models(self):
        errors = []

        if os.path.exists(MODEL_PATH):
            try:
                self.model = tf.keras.models.load_model(MODEL_PATH)
                self.info_label.config(
                    text=f"Model loaded • Input {self.model.input_shape}"
                )
            except Exception as e:
                errors.append(f"Model error: {e}")
        else:
            errors.append(f"Model not found:\n{MODEL_PATH}")

        if os.path.exists(HAND_LANDMARKER_PATH):
            try:
                base_opts = mp_python.BaseOptions(
                    model_asset_path=HAND_LANDMARKER_PATH
                )
                opts = vision.HandLandmarkerOptions(
                    base_options=base_opts,
                    num_hands=1,
                    running_mode=vision.RunningMode.VIDEO,
                )
                self.landmarker = vision.HandLandmarker.create_from_options(opts)
            except Exception as e:
                errors.append(f"MediaPipe error: {e}")
        else:
            errors.append(
                f"hand_landmarker.task not found:\n{HAND_LANDMARKER_PATH}\n"
                "Download from developers.google.com/mediapipe"
            )

        if errors:
            self.info_label.config(text="\n".join(errors), fg=ACCENT_RED)

    # ─────────────────────────────────────────────
    # START / STOP
    # ─────────────────────────────────────────────
    def _start(self):
        if self.running:
            return
        if self.model is None or self.landmarker is None:
            self.info_label.config(
                text="Cannot start — model or landmarker missing", fg=ACCENT_RED
            )
            return

        self.cap = cv2.VideoCapture(CAMERA_INDEX)
        if not self.cap.isOpened():
            self.info_label.config(text="Cannot open webcam", fg=ACCENT_RED)
            return

        self.running = True
        self.start_btn.config(state="disabled")
        self.stop_btn.config(state="normal")
        self.status_label.config(text="LIVE", fg=ACCENT_GREEN)
        self.status_dot.config(fg=ACCENT_GREEN)
        self.info_label.config(text="Camera active", fg=TEXT_DIM)

        self.recent_preds.clear()
        self.hold_letter = None
        self.hold_start = 0.0
        self.last_confirmed_key = None

        self.thread = threading.Thread(target=self._camera_loop, daemon=True)
        self.thread.start()

    def _stop(self):
        self.running = False
        if self.cap:
            self.cap.release()
            self.cap = None

        self.start_btn.config(state="normal")
        self.stop_btn.config(state="disabled")
        self.status_label.config(text="OFFLINE", fg=ACCENT_RED)
        self.status_dot.config(fg=ACCENT_RED)
        self.info_label.config(text="Camera stopped", fg=TEXT_DIM)
        self.letter_label.config(text="—", fg=TEXT_DIM)
        self.conf_label.config(text="Confidence: —")
        self.hold_label.config(text="Hold a sign for 3 seconds to confirm")
        self.progress_bar.place_configure(width=0)
        self._highlight_alpha(None)

    # ─────────────────────────────────────────────
    # CAMERA LOOP (background thread)
    # ─────────────────────────────────────────────
    def _camera_loop(self):
        while self.running:
            if not self.cap or not self.cap.isOpened():
                break

            ret, frame = self.cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            h, w = frame.shape[:2]
            display = frame.copy()

            # MediaPipe detection
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            ts = int(time.time() * 1000)

            try:
                result = self.landmarker.detect_for_video(mp_img, ts)
            except Exception:
                continue

            letter = None
            conf = 0.0
            crop_bgr = None

            if result.hand_landmarks:
                lms = result.hand_landmarks[0]
                x1, y1, x2, y2 = get_hand_bbox(lms, w, h, padding=PADDING)
                x1, y1, x2, y2 = make_square_bbox(x1, y1, x2, y2, w, h)

                # Draw bounding box on the display frame
                cv2.rectangle(display, (x1, y1), (x2, y2), (0, 230, 118), 2)

                crop_bgr = frame[y1:y2, x1:x2]

                if crop_bgr.size > 0:
                    inp = preprocess_crop(crop_bgr)
                    preds = self.model.predict(inp, verbose=0)[0]
                    idx = int(np.argmax(preds))
                    conf = float(preds[idx])
                    letter = CLASS_NAMES[idx]

            # Update GUI from the main thread
            self.root.after(0, self._update_gui, display, crop_bgr, letter, conf)
            time.sleep(0.03)  # ~30 FPS cap

    # ─────────────────────────────────────────────
    # GUI UPDATE (main thread)
    # ─────────────────────────────────────────────
    def _update_gui(self, display_bgr, crop_bgr, letter, conf):
        if not self.running:
            return

        # Camera feed — crop-to-fill (no black bars)
        cam_rgb = cv2.cvtColor(display_bgr, cv2.COLOR_BGR2RGB)
        lw = self.cam_label.winfo_width()
        lh = self.cam_label.winfo_height()
        if lw > 1 and lh > 1:
            src_h, src_w = cam_rgb.shape[:2]
            # Scale so smallest dimension fills the label (crop overflow)
            scale = max(lw / src_w, lh / src_h)
            new_w = int(src_w * scale)
            new_h = int(src_h * scale)
            cam_rgb = cv2.resize(cam_rgb, (new_w, new_h), interpolation=cv2.INTER_AREA)
            # Center-crop to label size
            y_off = (new_h - lh) // 2
            x_off = (new_w - lw) // 2
            cam_rgb = cam_rgb[y_off:y_off + lh, x_off:x_off + lw]
        cam_img = ImageTk.PhotoImage(Image.fromarray(cam_rgb))
        self.cam_label.config(image=cam_img)
        self.cam_label._img = cam_img  # prevent garbage collection

        # Hand crop
        if crop_bgr is not None and crop_bgr.size > 0:
            c_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
            c_rgb = cv2.resize(c_rgb, (180, 180), interpolation=cv2.INTER_AREA)
            c_img = ImageTk.PhotoImage(Image.fromarray(c_rgb))
            self.crop_label.config(image=c_img)
            self.crop_label._img = c_img

        # Prediction
        if letter and letter != "nothing" and conf >= CONFIDENCE_THRESHOLD:
            disp = letter if letter not in ("del", "space") else letter.upper()
            self.letter_label.config(text=disp, fg=ACCENT_GREEN)

            clr = (ACCENT_GREEN if conf > 0.85
                   else ACCENT_AMBER if conf > 0.70
                   else ACCENT_RED)
            self.conf_label.config(text=f"Confidence: {conf:.1%}", fg=clr)
            self._highlight_alpha(letter if len(letter) == 1 else None)
            self._update_hold(letter)
        else:
            self.letter_label.config(text="—", fg=TEXT_DIM)
            self.conf_label.config(
                text=f"Confidence: {conf:.1%}" if conf > 0 else "Confidence: —",
                fg=TEXT_DIM,
            )
            self._highlight_alpha(None)
            self._reset_hold()

    # ─────────────────────────────────────────────
    # HOLD-TO-CONFIRM
    # ─────────────────────────────────────────────
    def _update_hold(self, letter):
        now = time.time()

        if self.hold_letter == letter:
            elapsed = now - self.hold_start
            progress = min(elapsed / LETTER_HOLD_SECONDS, 1.0)

            bar_w = int(self.progress_frame.winfo_width() * progress)
            self.progress_bar.place_configure(width=max(bar_w, 0))
            self.progress_bar.config(
                bg="#00FF88" if progress >= 1.0 else ACCENT_GREEN
            )

            pct = int(progress * 100)
            if progress < 1.0:
                self.hold_label.config(text=f"Hold steady… {pct}%", fg=TEXT_DIM)
            else:
                self.hold_label.config(text="✓ Confirmed!", fg=ACCENT_GREEN)

            key = f"{letter}-{self.hold_start}"
            if progress >= 1.0 and self.last_confirmed_key != key:
                self.last_confirmed_key = key
                self._confirm_letter(letter)
                self.hold_start = now  # allow repeat
        else:
            self.hold_letter = letter
            self.hold_start = now
            self.progress_bar.place_configure(width=0)
            self.hold_label.config(
                text="Hold a sign for 3 seconds to confirm", fg=TEXT_DIM
            )

    def _reset_hold(self):
        self.hold_letter = None
        self.hold_start = 0.0
        self.progress_bar.place_configure(width=0)
        self.hold_label.config(
            text="Hold a sign for 3 seconds to confirm", fg=TEXT_DIM
        )

    def _confirm_letter(self, letter):
        if letter == "del":
            self.sentence = self.sentence[:-1]
        elif letter == "space":
            self.sentence += " "
        else:
            self.sentence += letter
        self.sentence_label.config(text=self.sentence)

    # ─────────────────────────────────────────────
    # SENTENCE ACTIONS
    # ─────────────────────────────────────────────
    def _delete_last(self):
        self.sentence = self.sentence[:-1]
        self.sentence_label.config(text=self.sentence)

    def _clear_sentence(self):
        self.sentence = ""
        self.sentence_label.config(text="")

    # ─────────────────────────────────────────────
    # ALPHABET HIGHLIGHT
    # ─────────────────────────────────────────────
    def _highlight_alpha(self, ch):
        for key, val in self.alpha_labels.items():
            cell, letter_lbl = val
            if key == ch:
                cell.config(bg=ACCENT_GREEN, highlightbackground=ACCENT_GREEN)
                letter_lbl.config(bg=ACCENT_GREEN, fg=BG_DARK)
            else:
                cell.config(bg=BG_PANEL, highlightbackground=BORDER)
                letter_lbl.config(bg=BG_PANEL, fg=TEXT_DIM)

    # ─────────────────────────────────────────────
    # CLEANUP
    # ─────────────────────────────────────────────
    def on_close(self):
        self.running = False
        if self.cap:
            self.cap.release()
        self.root.destroy()


# =============================================================================
# RUN
# =============================================================================
def main():
    root = tk.Tk()
    root.geometry("1200x750")
    app = ASLApp(root)
    root.protocol("WM_DELETE_WINDOW", app.on_close)
    root.mainloop()


if __name__ == "__main__":
    main()