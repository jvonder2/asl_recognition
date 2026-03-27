"""
ASL Live Inference Script
=========================
Uses MediaPipe Hand Landmarker to detect hands via webcam,
crops and preprocesses the hand region, then classifies it
with a trained Keras CNN.

Controls:
    q = quit
    p = pause
    s = save current crop
    d = toggle debug window (shows preprocessed 96x96 input)
    +/- = adjust padding live

Requirements:
    pip install tensorflow opencv-python mediapipe numpy
"""

import os
import time
from collections import deque

import cv2
import numpy as np
import tensorflow as tf
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


# =============================================================================
# CONFIG
# =============================================================================
MODEL_PATH = "C:\\Users\\jvond\\ML_Project\\asl_finetuned\\asl_model_finetuned.keras"
HAND_LANDMARKER_PATH = "hand_landmarker.task"

IMG_SIZE = 96
CAMERA_INDEX = 0
CONFIDENCE_THRESHOLD = 0.70
SHOW_WINDOWS = True

# Starting padding around the hand bounding box (in pixels).
# This is deliberately large so the crop framing is closer to the
# Kaggle training images, where the hand fills roughly 60-70% of the frame
# with background visible around it.  Adjust live with +/- keys.
PADDING = 80

# The model already contains a Rescaling(1./255) layer,
# so we should NOT normalise again here.
NORMALIZE_INPUT = False

CLASS_NAMES = [
    "A", "B", "C", "D", "E", "F", "G", "H", "I", "J",
    "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T",
    "U", "V", "W", "X", "Y", "Z", "del", "space", "nothing"
]

# Number of recent predictions to keep for smoothing / stability
PREDICTION_HISTORY = 5


# =============================================================================
# HELPERS
# =============================================================================
def make_square_bbox(
    x_min: int, y_min: int, x_max: int, y_max: int,
    frame_width: int, frame_height: int,
) -> tuple[int, int, int, int]:
    """
    Expand a bounding box into a square, centered on the original box.

    If the square would extend beyond the frame, it shifts (rather than
    shrinks) so the crop stays as large as possible.  It only shrinks
    if the frame itself is smaller than the desired side length.
    """
    box_w = x_max - x_min
    box_h = y_max - y_min
    side = max(box_w, box_h)

    cx = (x_min + x_max) // 2
    cy = (y_min + y_max) // 2

    half = side // 2

    # Start centered
    new_x_min = cx - half
    new_y_min = cy - half
    new_x_max = cx + half
    new_y_max = cy + half

    # Shift (not shrink) if we overflow the frame edges
    if new_x_min < 0:
        new_x_max -= new_x_min  # shift right
        new_x_min = 0
    if new_y_min < 0:
        new_y_max -= new_y_min  # shift down
        new_y_min = 0
    if new_x_max > frame_width:
        new_x_min -= (new_x_max - frame_width)  # shift left
        new_x_max = frame_width
    if new_y_max > frame_height:
        new_y_min -= (new_y_max - frame_height)  # shift up
        new_y_max = frame_height

    # Final clamp (only matters if the frame is smaller than `side`)
    new_x_min = max(new_x_min, 0)
    new_y_min = max(new_y_min, 0)
    new_x_max = min(new_x_max, frame_width)
    new_y_max = min(new_y_max, frame_height)

    return new_x_min, new_y_min, new_x_max, new_y_max


def get_hand_bbox(
    hand_landmarks,
    frame_width: int,
    frame_height: int,
    padding: int = 80,
) -> tuple[int, int, int, int]:
    """Convert MediaPipe normalised landmarks into a padded pixel bounding box."""
    xs = [lm.x for lm in hand_landmarks]
    ys = [lm.y for lm in hand_landmarks]

    x_min = max(int(min(xs) * frame_width) - padding, 0)
    y_min = max(int(min(ys) * frame_height) - padding, 0)
    x_max = min(int(max(xs) * frame_width) + padding, frame_width)
    y_max = min(int(max(ys) * frame_height) + padding, frame_height)

    return x_min, y_min, x_max, y_max


def preprocess_hand_crop(crop_bgr: np.ndarray) -> np.ndarray:
    """
    Resize and format a cropped hand image for the CNN.

    Steps:
        1. Resize to IMG_SIZE x IMG_SIZE  (matching training resolution)
        2. Convert BGR → RGB            (Keras models expect RGB)
        3. Cast to float32
        4. Optionally divide by 255     (only if model has no built-in Rescaling)
        5. Add batch dimension
    """
    resized = cv2.resize(crop_bgr, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    arr = rgb.astype(np.float32)

    if NORMALIZE_INPUT:
        arr /= 255.0

    return np.expand_dims(arr, axis=0)


def predict_sign(model, crop_bgr: np.ndarray):
    """Run inference and return (label, confidence, raw_predictions)."""
    x = preprocess_hand_crop(crop_bgr)
    preds = model.predict(x, verbose=0)[0]

    class_idx = int(np.argmax(preds))
    confidence = float(preds[class_idx])
    label = CLASS_NAMES[class_idx]

    # Print top-5 to terminal for debugging
    top5 = np.argsort(preds)[-5:][::-1]
    print("\nTop 5 predictions:")
    for i in top5:
        print(f"  {CLASS_NAMES[i]:>8s}: {preds[i]:.4f}")

    return label, confidence, preds


# =============================================================================
# MAIN LOOP
# =============================================================================
def main():
    # ------ Validate files ------
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")
    if not os.path.exists(HAND_LANDMARKER_PATH):
        raise FileNotFoundError(
            f"MediaPipe task file not found: {HAND_LANDMARKER_PATH}\n"
            "Download hand_landmarker.task from the MediaPipe docs."
        )

    # ------ Load model ------
    print("Loading Keras model...")
    model = tf.keras.models.load_model(MODEL_PATH)
    print(f"  Input shape : {model.input_shape}")
    print(f"  Output shape: {model.output_shape}")

    # ------ Load MediaPipe ------
    print("Loading MediaPipe Hand Landmarker...")
    base_options = python.BaseOptions(model_asset_path=HAND_LANDMARKER_PATH)
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        num_hands=1,
        running_mode=vision.RunningMode.VIDEO,
    )
    landmarker = vision.HandLandmarker.create_from_options(options)

    # ------ Open webcam ------
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam.")

    print("\nWebcam started.")
    print("Controls:")
    print("  q = quit")
    print("  p = pause / resume")
    print("  s = save current crop")
    print("  d = toggle debug view (preprocessed 96x96)")
    print("  + = increase padding   - = decrease padding\n")

    paused = False
    show_debug = False
    last_crop = None
    last_printed_text = ""
    recent_predictions: deque = deque(maxlen=PREDICTION_HISTORY)
    padding = PADDING  # mutable copy so we can adjust at runtime

    while True:
        # ---- Handle keys (always, even when paused) ----
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break
        elif key == ord("p"):
            paused = not paused
            print("⏸  Paused." if paused else "▶  Resumed.")
            continue
        elif key == ord("s"):
            if last_crop is not None:
                fname = f"saved_crop_{int(time.time() * 1000)}.png"
                cv2.imwrite(fname, last_crop)
                print(f"Saved crop → {fname}")
            else:
                print("No crop to save yet.")
            continue
        elif key == ord("d"):
            show_debug = not show_debug
            if not show_debug:
                cv2.destroyWindow("Debug: Model Input")
            print(f"Debug view {'ON' if show_debug else 'OFF'}")
            continue
        elif key in (ord("+"), ord("=")):
            padding = min(padding + 10, 200)
            print(f"Padding → {padding}")
            continue
        elif key == ord("-"):
            padding = max(padding - 10, 10)
            print(f"Padding → {padding}")
            continue

        if paused:
            continue

        # ---- Read frame ----
        ret, frame = cap.read()
        if not ret:
            print("Failed to read frame.")
            break

        frame = cv2.flip(frame, 1)
        display = frame.copy()
        h, w = frame.shape[:2]

        # ---- Detect hand ----
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        timestamp_ms = int(time.time() * 1000)
        result = landmarker.detect_for_video(mp_image, timestamp_ms)

        prediction_text = "No hand detected"

        if result.hand_landmarks:
            landmarks = result.hand_landmarks[0]

            # Build a padded, square bounding box
            x1, y1, x2, y2 = get_hand_bbox(landmarks, w, h, padding=padding)
            x1, y1, x2, y2 = make_square_bbox(x1, y1, x2, y2, w, h)

            # Draw box on display frame
            cv2.rectangle(display, (x1, y1), (x2, y2), (0, 255, 0), 2)

            hand_crop = frame[y1:y2, x1:x2]

            if hand_crop.size > 0:
                last_crop = hand_crop.copy()

                # Optional: show the exact 96x96 image the model sees
                if show_debug:
                    debug_img = cv2.resize(hand_crop, (IMG_SIZE, IMG_SIZE))
                    # Scale up so it's easier to see in the window
                    debug_display = cv2.resize(debug_img, (400, 400),
                                               interpolation=cv2.INTER_NEAREST)
                    cv2.imshow("Debug: Model Input", debug_display)

                label, confidence, preds = predict_sign(model, hand_crop)

                if confidence >= CONFIDENCE_THRESHOLD:
                    recent_predictions.append((label, confidence))

                    # Majority-vote over recent window
                    stable_label = max(
                        set(l for l, _ in recent_predictions),
                        key=lambda lbl: sum(
                            1 for l, _ in recent_predictions if l == lbl
                        ),
                    )
                    avg_conf = sum(c for _, c in recent_predictions) / len(
                        recent_predictions
                    )
                    prediction_text = f"{stable_label} ({avg_conf:.2f})"
                else:
                    recent_predictions.clear()
                    prediction_text = f"Low confidence ({confidence:.2f})"
            else:
                recent_predictions.clear()
        else:
            recent_predictions.clear()

        # ---- Terminal output (only on change) ----
        if prediction_text != last_printed_text:
            print(prediction_text)
            last_printed_text = prediction_text

        # ---- On-screen overlay ----
        if SHOW_WINDOWS:
            cv2.putText(display, prediction_text, (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.putText(display, f"Threshold: {CONFIDENCE_THRESHOLD:.2f}",
                        (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
            cv2.putText(display, f"Padding: {padding}",
                        (20, 115), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

            if paused:
                cv2.putText(display, "PAUSED", (20, 155),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

            cv2.imshow("ASL Hand Sign Predictor", display)

            if last_crop is not None:
                cv2.imshow("Hand Crop", last_crop)

    cap.release()
    cv2.destroyAllWindows()
    print("Closed cleanly.")


if __name__ == "__main__":
    main()