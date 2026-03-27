"""
ASL Data Collection Script
==========================

Collects cropped hand images for ASL symbols using MediaPipe hand tracking.

Features:
- Saves 200x200 cropped hand images
- Organizes by class folder
- Auto-collects N images per symbol
- Lets you choose output directory
- Uses padded square crop similar to your inference pipeline

Controls:
    g = start / resume collecting for current symbol
    p = pause collecting
    n = skip to next symbol
    r = reset current symbol folder count to 0 for this session
    q = quit

Requirements:
    pip install opencv-python mediapipe numpy
"""

import os
import time
from pathlib import Path

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


# =============================================================================
# CONFIG
# =============================================================================
HAND_LANDMARKER_PATH = "hand_landmarker.task"
CAMERA_INDEX = 0

IMG_SIZE = 200
PADDING = 80
SAVE_EVERY_N_SECONDS = 0.20   # save one image every 0.20 sec while collecting
TARGET_PER_SYMBOL = 250

# Change this if you want a different default output location
OUTPUT_ROOT = r"C:\Users\jvond\ML_Project\my_asl_dataset"

# Put whichever labels you want to collect here.
# You can leave out "nothing" for now if you're only collecting actual symbols.
SYMBOLS = [
    "A", "B", "C", "D", "E", "F", "G", "H", "I", "J",
    "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T",
    "U", "V", "W", "X", "Y", "Z", "del", "space"
]


# =============================================================================
# HELPERS
# =============================================================================
def make_square_bbox(
    x_min: int, y_min: int, x_max: int, y_max: int,
    frame_width: int, frame_height: int,
) -> tuple[int, int, int, int]:
    box_w = x_max - x_min
    box_h = y_max - y_min
    side = max(box_w, box_h)

    cx = (x_min + x_max) // 2
    cy = (y_min + y_max) // 2
    half = side // 2

    new_x_min = cx - half
    new_y_min = cy - half
    new_x_max = cx + half
    new_y_max = cy + half

    if new_x_min < 0:
        new_x_max -= new_x_min
        new_x_min = 0
    if new_y_min < 0:
        new_y_max -= new_y_min
        new_y_min = 0
    if new_x_max > frame_width:
        new_x_min -= (new_x_max - frame_width)
        new_x_max = frame_width
    if new_y_max > frame_height:
        new_y_min -= (new_y_max - frame_height)
        new_y_max = frame_height

    new_x_min = max(new_x_min, 0)
    new_y_min = max(new_y_min, 0)
    new_x_max = min(new_x_max, frame_width)
    new_y_max = min(new_y_max, frame_height)

    return new_x_min, new_y_min, new_x_max, new_y_max


def get_hand_bbox(hand_landmarks, frame_width: int, frame_height: int, padding: int):
    xs = [lm.x for lm in hand_landmarks]
    ys = [lm.y for lm in hand_landmarks]

    x_min = max(int(min(xs) * frame_width) - padding, 0)
    y_min = max(int(min(ys) * frame_height) - padding, 0)
    x_max = min(int(max(xs) * frame_width) + padding, frame_width)
    y_max = min(int(max(ys) * frame_height) + padding, frame_height)

    return x_min, y_min, x_max, y_max


def preprocess_crop_for_saving(crop_bgr: np.ndarray) -> np.ndarray:
    """
    Match your training data format:
    - square padded crop
    - resized to 200x200
    - saved as BGR image with OpenCV
    """
    resized = cv2.resize(crop_bgr, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
    return resized


def ensure_dirs(root: Path, symbols: list[str]) -> None:
    root.mkdir(parents=True, exist_ok=True)
    for symbol in symbols:
        (root / symbol).mkdir(parents=True, exist_ok=True)


def count_existing_images(folder: Path) -> int:
    valid_exts = {".jpg", ".jpeg", ".png"}
    return sum(1 for p in folder.iterdir() if p.suffix.lower() in valid_exts)


def save_crop(folder: Path, crop_bgr: np.ndarray, symbol: str) -> str:
    ts = int(time.time() * 1000)
    filename = folder / f"{symbol}_{ts}.jpg"
    cv2.imwrite(str(filename), crop_bgr)
    return str(filename)


# =============================================================================
# MAIN
# =============================================================================
def main():
    output_root = Path(OUTPUT_ROOT)
    ensure_dirs(output_root, SYMBOLS)

    if not os.path.exists(HAND_LANDMARKER_PATH):
        raise FileNotFoundError(
            f"MediaPipe task file not found: {HAND_LANDMARKER_PATH}\n"
            "Download hand_landmarker.task from the MediaPipe docs."
        )

    base_options = python.BaseOptions(model_asset_path=HAND_LANDMARKER_PATH)
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        num_hands=1,
        running_mode=vision.RunningMode.VIDEO,
    )
    landmarker = vision.HandLandmarker.create_from_options(options)

    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam.")

    print("\nASL Data Collection Started")
    print(f"Output root: {output_root}")
    print(f"Target per symbol: {TARGET_PER_SYMBOL}")
    print("\nControls:")
    print("  g = start/resume collecting")
    print("  p = pause collecting")
    print("  n = skip to next symbol")
    print("  r = reset current symbol count for this session")
    print("  q = quit\n")

    symbol_index = 0
    collecting = False
    last_save_time = 0.0
    session_counts = {symbol: 0 for symbol in SYMBOLS}

    while True:
        if symbol_index >= len(SYMBOLS):
            print("Finished all symbols.")
            break

        current_symbol = SYMBOLS[symbol_index]
        current_folder = output_root / current_symbol

        existing_count = count_existing_images(current_folder)
        total_count = existing_count
        remaining = max(TARGET_PER_SYMBOL - total_count, 0)

        ret, frame = cap.read()
        if not ret:
            print("Failed to read frame.")
            break

        frame = cv2.flip(frame, 1)
        display = frame.copy()
        h, w = frame.shape[:2]

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        timestamp_ms = int(time.time() * 1000)
        result = landmarker.detect_for_video(mp_image, timestamp_ms)

        crop_preview = None
        hand_detected = False

        if result.hand_landmarks:
            hand_detected = True
            landmarks = result.hand_landmarks[0]

            x1, y1, x2, y2 = get_hand_bbox(landmarks, w, h, padding=PADDING)
            x1, y1, x2, y2 = make_square_bbox(x1, y1, x2, y2, w, h)

            cv2.rectangle(display, (x1, y1), (x2, y2), (0, 255, 0), 2)

            crop = frame[y1:y2, x1:x2]
            if crop.size > 0:
                crop_preview = preprocess_crop_for_saving(crop)

                if collecting and remaining > 0:
                    now = time.time()
                    if now - last_save_time >= SAVE_EVERY_N_SECONDS:
                        save_crop(current_folder, crop_preview, current_symbol)
                        session_counts[current_symbol] += 1
                        last_save_time = now

                        total_count = count_existing_images(current_folder)
                        remaining = max(TARGET_PER_SYMBOL - total_count, 0)

                        if remaining == 0:
                            print(f"Done with {current_symbol}")
                            collecting = False
                            symbol_index += 1
                            continue

        # Overlay info
        status_text = "COLLECTING" if collecting else "PAUSED"
        cv2.putText(display, f"Symbol: {current_symbol}", (20, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
        cv2.putText(display, f"Status: {status_text}", (20, 75),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0) if collecting else (0, 0, 255), 2)
        cv2.putText(display, f"Saved: {total_count}/{TARGET_PER_SYMBOL}", (20, 110),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(display, f"Hand detected: {'Yes' if hand_detected else 'No'}", (20, 145),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(display, "g=start  p=pause  n=next  r=reset-session  q=quit", (20, h - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)

        cv2.imshow("ASL Collector", display)

        if crop_preview is not None:
            crop_big = cv2.resize(crop_preview, (400, 400), interpolation=cv2.INTER_NEAREST)
            cv2.imshow("Saved Crop Preview (200x200 source)", crop_big)

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break
        elif key == ord("g"):
            collecting = True
            print(f"Collecting for {current_symbol}...")
        elif key == ord("p"):
            collecting = False
            print("Paused.")
        elif key == ord("n"):
            collecting = False
            print(f"Skipping {current_symbol}")
            symbol_index += 1
        elif key == ord("r"):
            collecting = False
            print(f"Reset requested for session count of {current_symbol}.")
            # Optional: delete existing images in folder for a true reset
            for file in current_folder.glob("*"):
                if file.suffix.lower() in {".jpg", ".jpeg", ".png"}:
                    file.unlink()
            session_counts[current_symbol] = 0
            print(f"Cleared files in {current_folder}")

    cap.release()
    cv2.destroyAllWindows()
    print("Closed cleanly.")


if __name__ == "__main__":
    main()