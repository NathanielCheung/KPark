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
"""

import os
import time
from typing import Optional

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
FRAME_INTERVAL_SECONDS = float(os.environ.get("FRAME_INTERVAL_SECONDS", "2.0"))

# 6) Only POST when availableSpots changes by at least this many spots.
MIN_CHANGE_TO_POST = int(os.environ.get("MIN_CHANGE_TO_POST", "1"))


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


def count_cars_in_frame(model: "YOLO", frame) -> int:
    """
    Run the detection model on a single frame and return an approximate car count.
    """
    try:
        results = model(frame, verbose=False)
    except Exception as e:
        print(f"[WARN] Model inference failed: {e}")
        return 0

    if not results:
        return 0

    result = results[0]
    car_count = 0

    # COCO class index for "car" is 2 in many YOLO variants, but here we use names.
    # We check the 'names' mapping to be safe.
    names = result.names

    boxes = result.boxes
    for box in boxes:
        cls_id = int(box.cls[0])
        class_name = names.get(cls_id, "")
        if class_name in {"car", "truck", "bus"}:
            car_count += 1

    return car_count


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

            detected_cars = count_cars_in_frame(model, frame)
            available_spots = compute_available_spots(detected_cars)

            print(
                f"[INFO] Detected cars: {detected_cars}, "
                f"computed availableSpots: {available_spots}"
            )

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

