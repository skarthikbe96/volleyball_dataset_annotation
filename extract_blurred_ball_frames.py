import cv2
import numpy as np
import os
from statistics import median

# -----------------------------
# CONFIGURATION
# -----------------------------

VIDEO_PATH = "/home/rebellion/mobile_robotics/volleyball_dataset/position_C_dataset_1.mp4"
OUTPUT_DIR = "dataset_1_blurred_ball_frames"
DEBUG_DIR = "dataset_1_debug_blur"

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(DEBUG_DIR, exist_ok=True)

# Ball detection config – same as before (you said detection works)
BALL_HSV_LOWER = np.array([0, 0, 150])
BALL_HSV_UPPER = np.array([180, 60, 255])
MIN_BALL_RADIUS = 4
MAX_BALL_RADIUS = 80

# How many of the blurriest frames to keep:
# 60.0 = keep blurriest 60% (less strict than 30%)
BLUR_PERCENTILE = 60.0

# Also save ±N frames around each blurred frame (optional)
SAVE_NEIGHBORS = 0

NUM_DEBUG_FRAMES = 30


# -----------------------------
# BALL DETECTION
# -----------------------------

def detect_ball(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, BALL_HSV_LOWER, BALL_HSV_UPPER)

    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.erode(mask, kernel, iterations=1)
    mask = cv2.dilate(mask, kernel, iterations=2)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    best_candidate = None
    best_radius = 0

    for cnt in contours:
        (x_c, y_c), radius = cv2.minEnclosingCircle(cnt)
        radius = float(radius)

        if radius < MIN_BALL_RADIUS or radius > MAX_BALL_RADIUS:
            continue

        area = cv2.contourArea(cnt)
        if area <= 0:
            continue

        perimeter = cv2.arcLength(cnt, True)
        if perimeter == 0:
            continue
        circularity = 4.0 * np.pi * area / (perimeter ** 2)

        if circularity < 0.2:  # loose circularity
            continue

        if radius > best_radius:
            best_radius = radius
            best_candidate = (x_c, y_c, radius)

    if best_candidate is None:
        return False, None

    x_c, y_c, r = best_candidate
    x = int(x_c - r)
    y = int(y_c - r)
    w = int(2 * r)
    h = int(2 * r)

    h_frame, w_frame = frame.shape[:2]
    x = max(0, x)
    y = max(0, y)
    if x + w > w_frame:
        w = w_frame - x
    if y + h > h_frame:
        h = h_frame - y

    return True, (x, y, w, h)


# -----------------------------
# BLUR SCORE
# -----------------------------

def compute_blur_score(patch_bgr):
    if patch_bgr.size == 0:
        return None
    gray = cv2.cvtColor(patch_bgr, cv2.COLOR_BGR2GRAY)
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    return lap.var()


# -----------------------------
# MAIN
# -----------------------------

def main():
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print(f"Error opening video: {VIDEO_PATH}")
        return

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"Video: {VIDEO_PATH}")
    print(f"Frames: {frame_count}, FPS: {fps:.2f}")

    frames = []
    blur_scores = []
    ball_found_flags = []

    idx = 0
    detected_frames = 0

    # 1. Read video, detect ball, compute blur for each frame
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frames.append(frame)

        found, bbox = detect_ball(frame)
        if not found:
            ball_found_flags.append(False)
            blur_scores.append(None)
        else:
            ball_found_flags.append(True)
            detected_frames += 1

            x, y, w, h = bbox
            patch = frame[y:y+h, x:x+w]
            blur = compute_blur_score(patch)
            blur_scores.append(blur)

        idx += 1
        if idx % 100 == 0:
            print(f"Processed {idx}/{frame_count} frames...")

    cap.release()
    total_frames = len(frames)
    print(f"Finished reading video. Total frames: {total_frames}")
    print(f"Frames with ball detected: {detected_frames}")

    # 2. Blur stats
    valid_blurs = [b for b in blur_scores if b is not None]
    if not valid_blurs:
        print("No blur scores computed (ball may not be detected).")
        return

    print("Blur stats (Laplacian variance):")
    print(f"  min: {min(valid_blurs):.2f}")
    print(f"  max: {max(valid_blurs):.2f}")
    print(f"  median: {median(valid_blurs):.2f}")

    blur_array = np.array(valid_blurs)

    # Lower blur_score = more blurred.
    # BLUR_PERCENTILE% lowest values = blurriest BLUR_PERCENTILE% frames.
    blur_thresh = float(np.percentile(blur_array, BLUR_PERCENTILE))
    print(f"\nBlur threshold at {BLUR_PERCENTILE}th percentile (keep <= this): {blur_thresh:.2f}")

    # 3. Select frames with blur score <= threshold
    selected_indices = set()
    for i in range(total_frames):
        if not ball_found_flags[i]:
            continue
        b = blur_scores[i]
        if b is None:
            continue

        if b <= blur_thresh:
            for k in range(-SAVE_NEIGHBORS, SAVE_NEIGHBORS + 1):
                j = i + k
                if 0 <= j < total_frames:
                    selected_indices.add(j)

    selected_indices = sorted(selected_indices)
    print(f"\nSelected {len(selected_indices)} frames as blurred ball frames.")

    # 4. Save selected frames
    for idx in selected_indices:
        out_path = os.path.join(OUTPUT_DIR, f"frame_{idx:06d}.jpg")
        cv2.imwrite(out_path, frames[idx])

    print(f"Saved blurred-ball frames to: {OUTPUT_DIR}")

    # 5. Save some debug frames with blur value overlay
    debug_gap = max(1, total_frames // NUM_DEBUG_FRAMES)
    debug_saved = 0

    for i in range(0, total_frames, debug_gap):
        if debug_saved >= NUM_DEBUG_FRAMES:
            break

        frame = frames[i].copy()
        txt = ""
        if ball_found_flags[i] and blur_scores[i] is not None:
            txt = f"blur={blur_scores[i]:.1f}"
        else:
            txt = "ball: not detected"

        cv2.putText(frame, txt, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        out_debug = os.path.join(DEBUG_DIR, f"debug_{i:06d}.jpg")
        cv2.imwrite(out_debug, frame)
        debug_saved += 1

    print(f"Saved {debug_saved} debug frames to: {DEBUG_DIR}")


if __name__ == "__main__":
    main()
