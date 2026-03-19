"""
Wyze Cam v3 -> parking lot bridge (e.g. Ontario & Brock Lot).

This script runs on your local PC, connects to the Wyze camera video stream,
counts cars in view, and posts the resulting availableSpots to the existing
Express backend at /api/parking using a specified lotId
(for you: "ontario-brock-lot").

It is a prototype and aims to be simple to follow rather than perfectly optimized.

Prerequisites (install in a Python virtual environment is recommended):

    pip install opencv-python requests ultralytics

You also need a working RTSP (or similar) URL for your Wyze Cam v3 that your PC
can access on the local network.

If the car is visible but counts stay at 0, try (PowerShell):

    $env:YOLO_CONF="0.1"
    $env:YOLO_IMGSZ="1280"
    $env:DETECTION_DEBUG="1"
    $env:SHOW_PREVIEW="1"
    python wyze_beamish_detector.py

This lowers the confidence threshold, uses a larger inference size for small cars,
prints every detection label, and opens a window with bounding boxes.
"""

import os
import time
from typing import Any, Optional, Tuple

import cv2  # type: ignore
import requests  # type: ignore

try:
    from ultralytics import YOLO  # type: ignore
except ImportError:
    YOLO = None  # type: ignore


# ==== USER CONFIGURATION SECTION ====

# 1) Wyze Cam stream URL (RTSP or other supported by OpenCV).
#    Example (this will NOT be your exact URL):
#    rtsp://user:password@camera-ip/live
RTSP_URL = os.environ.get("WYZE_RTSP_URL", "rtsp://YOUR_WYZE_CAM_URL_HERE")

# 2) How many parking spots are visible in the camera view for this lot.
TOTAL_SPOTS = int(os.environ.get("WYZE_TOTAL_SPOTS", "3"))

# 3) Backend API endpoint for your existing Express server (/api/parking).
#    Typically: http://localhost:3001/api/parking
PARKING_API_URL = os.environ.get(
    "PARKING_API_URL", "http://localhost:3001/api/parking"
)

# 4) The lotId used by your app for this camera-controlled lot.
#    In your data, Ontario & Brock Lot is "ontario-brock-lot".
LOT_ID = os.environ.get("WYZE_LOT_ID", "ontario-brock-lot")

# 5) How often to analyze a frame and potentially send an update (in seconds).
FRAME_INTERVAL_SECONDS = float(os.environ.get("FRAME_INTERVAL_SECONDS", "1.0"))

# 6) Only POST when availableSpots changes by at least this many spots.
MIN_CHANGE_TO_POST = int(os.environ.get("MIN_CHANGE_TO_POST", "1"))

# 7) YOLO tuning (Wyze RTSP is often low-res / wide-angle — cars can be small).
#    Lower YOLO_CONF if you get "Detected cars: 0" with a visible car.
YOLO_CONF = float(os.environ.get("YOLO_CONF", "0.2"))
YOLO_IMGSZ = int(os.environ.get("YOLO_IMGSZ", "640"))  # try 1280 if car is tiny

# 8) Debug: print every box the model sees (label + confidence).
DETECTION_DEBUG = os.environ.get("DETECTION_DEBUG", "").lower() in (
    "1",
    "true",
    "yes",
)

# 9) Show an OpenCV window with boxes drawn (press q on that window may not exit; use Ctrl+C).
SHOW_PREVIEW = os.environ.get("SHOW_PREVIEW", "").lower() in ("1", "true", "yes")


# ==== END USER CONFIGURATION SECTION ====


def load_model() -> Optional["YOLO"]:
    """
    Load a YOLO model for car detection.

    Uses the default YOLOv8n model trained on COCO, which includes the "car" class.
    """
    if YOLO is None:
        print(
            "[ERROR] ultralytics not installed. "
            "Run 'pip install ultralytics' in your Python environment."
        )
        return None

    try:
        # This will download the model the first time it runs.
        model = YOLO("yolov8n.pt")
        return model
    except Exception as e:
        print(f"[ERROR] Failed to load YOLO model: {e}")
        return None


def count_cars_in_frame(model: "YOLO", frame) -> Tuple[int, Optional[Any]]:
    """
    Run the detection model on a single frame and return (car_count, annotated_frame).

    annotated_frame is a BGR numpy array (for preview) or None if SHOW_PREVIEW is False.
    """
    try:
        results = model(
            frame,
            verbose=False,
            conf=YOLO_CONF,
            imgsz=YOLO_IMGSZ,
        )
    except Exception as e:
        print(f"[WARN] Model inference failed: {e}")
        return 0, None

    if not results:
        return 0, None

    result = results[0]
    car_count = 0

    # COCO class index for "car" is 2 in many YOLO variants, but here we use names.
    # We check the 'names' mapping to be safe.
    names = result.names

    boxes = result.boxes
    if boxes is None or len(boxes) == 0:
        if DETECTION_DEBUG:
            print("[DEBUG] No detections this frame.")
        return 0, (result.plot() if SHOW_PREVIEW else None)

    for box in boxes:
        cls_id = int(box.cls[0])
        class_name = names.get(cls_id, "")
        conf = float(box.conf[0]) if box.conf is not None else 0.0
        if DETECTION_DEBUG:
            print(f"[DEBUG] detection: {class_name!r} conf={conf:.2f}")
        if class_name in {"car", "truck", "bus", "motorcycle"}:
            car_count += 1

    annotated = result.plot() if SHOW_PREVIEW else None
    return car_count, annotated


def clamp(value: int, min_value: int, max_value: int) -> int:
    return max(min_value, min(value, max_value))


def compute_available_spots(detected_cars: int) -> int:
    occupied = clamp(detected_cars, 0, TOTAL_SPOTS)
    available = TOTAL_SPOTS - occupied
    return available


def post_update_to_backend(available_spots: int) -> bool:
    payload = {"lotId": LOT_ID, "availableSpots": available_spots}
    try:
        resp = requests.post(PARKING_API_URL, json=payload, timeout=5)
        if 200 <= resp.status_code < 300:
            print(
                f"[OK] Posted update to backend: {payload} "
                f"(status {resp.status_code})"
            )
            return True
        print(
            f"[WARN] Backend responded with status {resp.status_code}: "
            f"{resp.text[:200]}"
        )
        return False
    except Exception as e:
        print(f"[ERROR] Failed to POST to backend: {e}")
        return False


def main() -> None:
    print("=== Wyze -> Beamish Munro Hall parking bridge ===")
    print(f"RTSP_URL          = {RTSP_URL}")
    print(f"PARKING_API_URL   = {PARKING_API_URL}")
    print(f"LOT_ID            = {LOT_ID}")
    print(f"TOTAL_SPOTS       = {TOTAL_SPOTS}")
    print(f"FRAME_INTERVAL_S  = {FRAME_INTERVAL_SECONDS}")
    print(f"YOLO_CONF         = {YOLO_CONF}")
    print(f"YOLO_IMGSZ        = {YOLO_IMGSZ}")
    print(f"DETECTION_DEBUG   = {DETECTION_DEBUG}")
    print(f"SHOW_PREVIEW      = {SHOW_PREVIEW}")

    if "YOUR_WYZE_CAM_URL_HERE" in RTSP_URL:
        print(
            "[ERROR] RTSP_URL not configured. "
            "Set WYZE_RTSP_URL env var or edit RTSP_URL in this script."
        )
        return

    model = load_model()
    if model is None:
        return

    cap = cv2.VideoCapture(RTSP_URL)
    if not cap.isOpened():
        print(
            "[ERROR] Could not open video stream. "
            "Check your RTSP URL and network connectivity."
        )
        return

    print("[INFO] Video stream opened. Press Ctrl+C to stop.")

    last_posted_available: Optional[int] = None
    last_frame_time = 0.0

    try:
        while True:
            now = time.time()
            if now - last_frame_time < FRAME_INTERVAL_SECONDS:
                # Sleep a bit to avoid busy waiting.
                time.sleep(0.1)
                continue

            last_frame_time = now

            ret, frame = cap.read()
            if not ret or frame is None:
                print("[WARN] Failed to read frame from camera.")
                time.sleep(1.0)
                continue

            detected_cars, preview = count_cars_in_frame(model, frame)
            available_spots = compute_available_spots(detected_cars)

            print(
                f"[INFO] Detected cars: {detected_cars}, "
                f"computed availableSpots: {available_spots}"
            )

            if SHOW_PREVIEW and preview is not None:
                cv2.imshow("wyze_detector (Ctrl+C to quit)", preview)
                cv2.waitKey(1)

            # Only POST when the available spots change significantly.
            if last_posted_available is None or abs(
                available_spots - last_posted_available
            ) >= MIN_CHANGE_TO_POST:
                if post_update_to_backend(available_spots):
                    last_posted_available = available_spots

    except KeyboardInterrupt:
        print("\n[INFO] Stopping (Ctrl+C pressed).")
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

