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
prints every detection label, and opens a preview window.

Timing: YOLO runs every FRAME_INTERVAL_SECONDS (default 2s). With SHOW_PREVIEW=1 and
PREVIEW_LIVE=1 (default), the window shows a live feed; counts on screen update each scan.
Set PREVIEW_LIVE=0 to show YOLO bounding boxes only when a scan runs.
"""

import os
import time
from typing import Any, List, Optional, Tuple

# Windows + RTSP: TCP + options that reduce H.264 "Missing reference picture" / decode errors
# when joining a Wyze/wyze-bridge stream mid-GOP. Override with OPENCV_FFMPEG_CAPTURE_OPTIONS.
if os.name == "nt" and "OPENCV_FFMPEG_CAPTURE_OPTIONS" not in os.environ:
    os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = (
        "rtsp_transport;tcp|"
        "fflags;discardcorrupt|"
        "max_delay;500000|"
        "stimeout;8000000"
    )

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

# 2) Total spaces in this lot (each detected vehicle counts as one occupied spot).
#    Ontario & Brock Lot in the app is 10 spaces — default matches that.
TOTAL_SPOTS = int(os.environ.get("WYZE_TOTAL_SPOTS", "10"))

# 3) Backend API endpoint for your existing Express server (/api/parking).
#    Typically: http://localhost:3001/api/parking
PARKING_API_URL = os.environ.get(
    "PARKING_API_URL", "http://localhost:3001/api/parking"
)

# 4) The lotId used by your app for this camera-controlled lot.
#    In your data, Ontario & Brock Lot is "ontario-brock-lot".
LOT_ID = os.environ.get("WYZE_LOT_ID", "ontario-brock-lot")

# 5) How often to run YOLO on a frame and potentially POST (in seconds).
#    Default 2.0 = "every other second" style cadence; override with FRAME_INTERVAL_SECONDS.
FRAME_INTERVAL_SECONDS = float(os.environ.get("FRAME_INTERVAL_SECONDS", "2.0"))

# 6) Only POST when availableSpots changes by at least this many spots.
MIN_CHANGE_TO_POST = int(os.environ.get("MIN_CHANGE_TO_POST", "1"))

# 7) YOLO tuning (Wyze RTSP is often low-res / wide-angle — cars can be small).
#    YOLO_CONF is passed to Ultralytics as the minimum confidence for a box to exist at all.
#    Values like 0.35–0.5 often drop *real* cars on Wyze; try 0.15–0.25 if you see cars but count stays 0.
YOLO_CONF = float(os.environ.get("YOLO_CONF", "0.2"))
YOLO_IMGSZ = int(os.environ.get("YOLO_IMGSZ", "640"))  # try 1280 if car is tiny

# Only count vehicle-class boxes if confidence >= this (optional).
# Use case: set YOLO_CONF=0.15 to "see" weak cars, VEHICLE_MIN_CONF=0.28 to avoid counting junk.
_vm = os.environ.get("VEHICLE_MIN_CONF", "").strip()
VEHICLE_MIN_CONF: Optional[float] = float(_vm) if _vm else None

# Restrict model output to COCO car, motorcycle, truck only (no bus) — less noise, slightly faster.
# Set YOLO_VEHICLE_ONLY=0 to run full COCO; we still only count/draw car, truck, motorcycle.
YOLO_VEHICLE_ONLY = os.environ.get("YOLO_VEHICLE_ONLY", "1").lower() not in (
    "0",
    "false",
    "no",
)
# YOLOv8 / COCO indices: car=2, motorcycle=3, truck=7 (bus omitted on purpose)
YOLO_VEHICLE_CLASS_IDS = [2, 3, 7]

# Optional: yolov8n.pt (fast) vs yolov8s.pt / yolov8m.pt (better on small cars, slower).
YOLO_MODEL_NAME = os.environ.get("YOLO_MODEL", "yolov8n.pt")

# 8) Debug: print every box the model sees (label + confidence).
DETECTION_DEBUG = os.environ.get("DETECTION_DEBUG", "").lower() in (
    "1",
    "true",
    "yes",
)

# 9) Show an OpenCV window (Ctrl+C in the terminal to quit).
SHOW_PREVIEW = os.environ.get("SHOW_PREVIEW", "").lower() in ("1", "true", "yes")

# 10) With SHOW_PREVIEW: stream live video and refresh the window often (recommended).
#     YOLO still runs only every FRAME_INTERVAL_SECONDS; overlay shows last counts.
#     Set PREVIEW_LIVE=0 to only open/update the window when YOLO runs (older behavior).
PREVIEW_LIVE = os.environ.get("PREVIEW_LIVE", "1").lower() not in ("0", "false", "no")

# When PREVIEW_LIVE=1, redraw car/truck/motorcycle boxes on each video frame using the last YOLO result
# (boxes match the last scan; smooth video + outlines). Set PREVIEW_BOX_OVERLAY_LIVE=0 for text only.
PREVIEW_BOX_OVERLAY_LIVE = os.environ.get("PREVIEW_BOX_OVERLAY_LIVE", "1").lower() not in (
    "0",
    "false",
    "no",
)

# 11) Max display refresh rate when PREVIEW_LIVE (limits CPU; RTSP read rate).
PREVIEW_MAX_FPS = float(os.environ.get("PREVIEW_MAX_FPS", "20"))

# 12) After opening RTSP, discard this many good frames to reach a stable keyframe (Wyze/FFmpeg).
WYZE_WARMUP_FRAMES = int(os.environ.get("WYZE_WARMUP_FRAMES", "45"))

# 14) Milliseconds to sleep between cap.read() calls (helps H.264 decoder; 0 = off).
WYZE_READ_PAUSE_MS = int(os.environ.get("WYZE_READ_PAUSE_MS", "15"))

# 15) Warmup treats frames with mean brightness below this as corrupt/black (lower for night video).
WYZE_MIN_FRAME_MEAN = float(os.environ.get("WYZE_MIN_FRAME_MEAN", "1.5"))

# 13) When the vehicle pass finds zero boxes, periodically run a low-threshold full-COCO
#     probe so you can see whether YOLO sees *anything* (wrong label vs dead stream).
WYZE_PROBE_WHEN_EMPTY = os.environ.get("WYZE_PROBE_WHEN_EMPTY", "1").lower() not in (
    "0",
    "false",
    "no",
)
WYZE_PROBE_INTERVAL_S = float(os.environ.get("WYZE_PROBE_INTERVAL_S", "12"))
WYZE_PROBE_CONF = float(os.environ.get("WYZE_PROBE_CONF", "0.08"))
WYZE_PROBE_MAX_LABELS = int(os.environ.get("WYZE_PROBE_MAX_LABELS", "12"))


# ==== END USER CONFIGURATION SECTION ====

# Only these COCO classes count toward occupancy and get drawn on the preview (no bus/plane/person/etc.).
COUNTED_VEHICLE_NAMES = frozenset({"car", "truck", "motorcycle"})


def open_rtsp_capture(url: str) -> cv2.VideoCapture:
    """
    Open RTSP with FFMPEG backend.

    Wyze H.264 often needs a *small multi-frame buffer* so P-frames still have their
    references. Buffer size 1 commonly triggers 'Missing reference picture' spam.
    Set WYZE_CAP_BUFFERSIZE=1 only if you want minimum latency and accept decode errors.
    """
    cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
    buf = os.environ.get("WYZE_CAP_BUFFERSIZE", "4").strip()
    if buf and buf.isdigit():
        try:
            cap.set(cv2.CAP_PROP_BUFFERSIZE, int(buf))
        except Exception:
            pass
    return cap


def _read_pause() -> None:
    if WYZE_READ_PAUSE_MS > 0:
        time.sleep(WYZE_READ_PAUSE_MS / 1000.0)


def read_frame(cap: cv2.VideoCapture) -> Tuple[bool, Any]:
    _read_pause()
    return cap.read()


def frame_brightness_ok(frame) -> Tuple[float, bool]:
    """Mean pixel value; False if frame looks black/empty."""
    if frame is None or frame.size == 0:
        return 0.0, False
    mean = float(frame.mean())
    return mean, mean >= WYZE_MIN_FRAME_MEAN


def warmup_rtsp_capture(cap: cv2.VideoCapture, n: int) -> Tuple[bool, Any]:
    """Read and discard frames until we collect n usable frames (helps after H264 errors)."""
    last = None
    good = 0
    attempts = 0
    max_attempts = max(n * 4, 80)
    while good < max(n, 0) and attempts < max_attempts:
        attempts += 1
        ret, frame = read_frame(cap)
        if not ret or frame is None:
            continue
        _mean_bgr, bright_ok = frame_brightness_ok(frame)
        if not bright_ok:
            continue
        good += 1
        last = frame
    return good > 0, last


def probe_all_coco_labels(model: "YOLO", frame) -> List[str]:
    """Low-threshold pass with no class filter — for debugging only."""
    lines: List[str] = []
    try:
        results = model(
            frame,
            verbose=False,
            conf=WYZE_PROBE_CONF,
            imgsz=min(YOLO_IMGSZ, 640),
        )
    except Exception as e:
        return [f"(probe failed: {e})"]
    if not results or results[0].boxes is None or len(results[0].boxes) == 0:
        return [
            "(no boxes at all — lower WYZE_PROBE_CONF, check RTSP, or frame may be black)"
        ]
    res = results[0]
    names = res.names
    for i, box in enumerate(res.boxes):
        if i >= WYZE_PROBE_MAX_LABELS:
            break
        cls_id = int(box.cls[0])
        cn = names.get(cls_id, "?")
        cf = float(box.conf[0]) if box.conf is not None else 0.0
        mark = " <-- counted vehicle" if cn in COUNTED_VEHICLE_NAMES else ""
        lines.append(f"  {cn!r} conf={cf:.2f}{mark}")
    return lines


def draw_counted_vehicle_boxes(frame, result, names: dict) -> Any:
    """
    Draw bounding boxes only for car / truck / motorcycle (same rules as counting).
    Does not use result.plot(), so birds, people, clocks, etc. never get boxes.
    """
    out = frame.copy()
    boxes = result.boxes
    if boxes is None or len(boxes) == 0:
        return out

    colors = {
        "car": (0, 255, 0),
        "truck": (0, 200, 255),
        "motorcycle": (255, 140, 0),
    }

    for box in boxes:
        cls_id = int(box.cls[0])
        class_name = names.get(cls_id, "")
        if class_name not in COUNTED_VEHICLE_NAMES:
            continue
        conf = float(box.conf[0]) if box.conf is not None else 0.0
        if VEHICLE_MIN_CONF is not None and conf < VEHICLE_MIN_CONF:
            continue

        xy = box.xyxy[0]
        try:
            x1, y1, x2, y2 = (int(xy[0]), int(xy[1]), int(xy[2]), int(xy[3]))
        except (TypeError, ValueError):
            continue

        color = colors.get(class_name, (0, 255, 0))
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        label = f"{class_name} {conf:.2f}"
        ly = max(18, y1 - 6)
        cv2.putText(
            out,
            label,
            (x1, ly),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 0, 0),
            3,
            cv2.LINE_AA,
        )
        cv2.putText(
            out,
            label,
            (x1, ly),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            1,
            cv2.LINE_AA,
        )
    return out


def draw_preview_overlay(
    frame,
    vehicles: int,
    available: int,
    infer_interval_s: float,
    just_ran_infer: bool,
) -> None:
    """Draw status text on a copy of the frame (BGR)."""
    lines = [
        f"Vehicles (YOLO): {vehicles}",
        f"Available: {available} / {TOTAL_SPOTS}",
        f"Scan every {infer_interval_s:.1f}s" + ("  [scan]" if just_ran_infer else ""),
    ]
    y = 28
    for line in lines:
        cv2.putText(
            frame,
            line,
            (10, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 0, 0),
            4,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            line,
            (10, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )
        y += 26


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
        # Downloads the first time (name from YOLO_MODEL, default yolov8n.pt).
        model = YOLO(YOLO_MODEL_NAME)
        return model
    except Exception as e:
        print(f"[ERROR] Failed to load YOLO model: {e}")
        return None


def count_cars_in_frame(model: "YOLO", frame) -> Tuple[int, Optional[Any], Optional[Any]]:
    """
    Run the detection model on a single frame.

    Returns (car_count, annotated_frame, yolo_result).
    - annotated_frame: BGR image with boxes only for car/truck/motorcycle, or None.
    - yolo_result: last Results object for live overlay (or None on failure).
    """
    try:
        infer_kw: dict = {
            "verbose": False,
            "conf": YOLO_CONF,
            "imgsz": YOLO_IMGSZ,
        }
        if YOLO_VEHICLE_ONLY:
            infer_kw["classes"] = YOLO_VEHICLE_CLASS_IDS
        results = model(frame, **infer_kw)
    except Exception as e:
        print(f"[WARN] Model inference failed: {e}")
        return 0, None, None

    if not results:
        return 0, None, None

    result = results[0]
    car_count = 0

    # COCO class index for "car" is 2 in many YOLO variants, but here we use names.
    # We check the 'names' mapping to be safe.
    names = result.names

    boxes = result.boxes
    if boxes is None or len(boxes) == 0:
        if DETECTION_DEBUG:
            print("[DEBUG] No detections this frame.")
        ann = draw_counted_vehicle_boxes(frame, result, names) if SHOW_PREVIEW else None
        return 0, ann, result

    for box in boxes:
        cls_id = int(box.cls[0])
        class_name = names.get(cls_id, "")
        conf = float(box.conf[0]) if box.conf is not None else 0.0

        if class_name not in COUNTED_VEHICLE_NAMES:
            if DETECTION_DEBUG:
                print(f"[DEBUG] detection: {class_name!r} conf={conf:.2f} [ignored class]")
            continue
        if VEHICLE_MIN_CONF is not None and conf < VEHICLE_MIN_CONF:
            if DETECTION_DEBUG:
                print(
                    f"[DEBUG] detection: {class_name!r} conf={conf:.2f} "
                    f"[below VEHICLE_MIN_CONF={VEHICLE_MIN_CONF}]"
                )
            continue
        if DETECTION_DEBUG:
            print(f"[DEBUG] detection: {class_name!r} conf={conf:.2f} [counted]")
        car_count += 1

    annotated = draw_counted_vehicle_boxes(frame, result, names) if SHOW_PREVIEW else None
    return car_count, annotated, result


def clamp(value: int, min_value: int, max_value: int) -> int:
    return max(min_value, min(value, max_value))


def compute_available_spots(vehicle_count: int) -> int:
    """availableSpots = TOTAL_SPOTS minus vehicles seen (capped so it never goes negative)."""
    occupied = clamp(vehicle_count, 0, TOTAL_SPOTS)
    return TOTAL_SPOTS - occupied


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
    print("=== Wyze -> Ontario & Brock Lot parking bridge ===")
    print(f"RTSP_URL          = {RTSP_URL}")
    print(f"PARKING_API_URL   = {PARKING_API_URL}")
    print(f"LOT_ID            = {LOT_ID}")
    print(f"TOTAL_SPOTS       = {TOTAL_SPOTS}")
    print(f"FRAME_INTERVAL_S  = {FRAME_INTERVAL_SECONDS}")
    print(f"YOLO_MODEL        = {YOLO_MODEL_NAME}")
    print(f"YOLO_CONF         = {YOLO_CONF}")
    print(f"YOLO_VEHICLE_ONLY = {YOLO_VEHICLE_ONLY}")
    print(f"VEHICLE_MIN_CONF  = {VEHICLE_MIN_CONF}")
    print(f"YOLO_IMGSZ        = {YOLO_IMGSZ}")
    print(f"DETECTION_DEBUG   = {DETECTION_DEBUG}")
    print(f"SHOW_PREVIEW      = {SHOW_PREVIEW}")
    print(f"PREVIEW_LIVE      = {PREVIEW_LIVE}")
    print(f"PREVIEW_BOX_OVERLAY = {PREVIEW_BOX_OVERLAY_LIVE}")
    print(f"PREVIEW_MAX_FPS   = {PREVIEW_MAX_FPS}")
    print(f"WYZE_WARMUP_FRAMES= {WYZE_WARMUP_FRAMES}")
    print(f"WYZE_READ_PAUSE_MS= {WYZE_READ_PAUSE_MS}")
    print(f"WYZE_CAP_BUFFERSIZE= {os.environ.get('WYZE_CAP_BUFFERSIZE', '4')}")
    print(f"WYZE_PROBE_EMPTY  = {WYZE_PROBE_WHEN_EMPTY} (interval {WYZE_PROBE_INTERVAL_S}s)")

    if "YOUR_WYZE_CAM_URL_HERE" in RTSP_URL:
        print(
            "[ERROR] RTSP_URL not configured. "
            "Set WYZE_RTSP_URL env var or edit RTSP_URL in this script."
        )
        return

    model = load_model()
    if model is None:
        return

    cap = open_rtsp_capture(RTSP_URL)
    if not cap.isOpened():
        print(
            "[ERROR] Could not open video stream. "
            "Check your RTSP URL and network connectivity."
        )
        return

    print("[INFO] Video stream opened. Warming up RTSP buffer…")
    warm_ok, warm_frame = warmup_rtsp_capture(cap, WYZE_WARMUP_FRAMES)
    if warm_ok and warm_frame is not None:
        h, w = warm_frame.shape[:2]
        mean_bgr, bright_ok = frame_brightness_ok(warm_frame)
        print(
            f"[INFO] After warmup: frame ~{w}x{h}, mean brightness={mean_bgr:.1f} "
            f"({'OK' if bright_ok else 'VERY DARK — stream may be wrong/black'})"
        )
    else:
        print(
            "[WARN] Warmup did not get valid frames — RTSP may be stalled. "
            "Check Wyze Bridge and URL."
        )

    print("[INFO] Press Ctrl+C to stop.")

    last_posted_available: Optional[int] = None
    last_infer_time = 0.0
    last_display_time = 0.0
    last_probe_time = 0.0
    min_display_dt = 1.0 / max(PREVIEW_MAX_FPS, 1.0)

    last_vehicles = 0
    last_available = TOTAL_SPOTS
    last_yolo_result: Optional[Any] = None

    try:
        while True:
            now = time.time()

            if SHOW_PREVIEW and PREVIEW_LIVE:
                # Smooth video: read often; run YOLO only every FRAME_INTERVAL_SECONDS.
                ret, frame = read_frame(cap)
                if not ret or frame is None:
                    print("[WARN] Failed to read frame from camera.")
                    time.sleep(1.0)
                    continue

                just_ran_infer = False
                if now - last_infer_time >= FRAME_INTERVAL_SECONDS:
                    last_infer_time = now
                    just_ran_infer = True

                    detected_cars, _preview, yolo_res = count_cars_in_frame(model, frame)
                    last_yolo_result = yolo_res
                    available_spots = compute_available_spots(detected_cars)
                    last_vehicles = detected_cars
                    last_available = available_spots

                    print(
                        f"[INFO] Detected vehicles: {detected_cars}, "
                        f"computed availableSpots: {available_spots}"
                    )

                    if (
                        detected_cars == 0
                        and WYZE_PROBE_WHEN_EMPTY
                        and (now - last_probe_time) >= WYZE_PROBE_INTERVAL_S
                    ):
                        last_probe_time = now
                        _, bright_ok = frame_brightness_ok(frame)
                        print(
                            "[PROBE] No vehicle boxes this scan. "
                            f"Frame OK={bright_ok}. "
                            f"Full-COCO hints (conf>={WYZE_PROBE_CONF}, no class filter):"
                        )
                        for line in probe_all_coco_labels(model, frame):
                            print(line)
                        print(
                            "[HINT] If you see vehicle labels above, lower YOLO_CONF or set "
                            "YOLO_VEHICLE_ONLY=0 temporarily. If probe shows nothing, lower "
                            "WYZE_PROBE_CONF or fix RTSP (TCP is set on Windows by default)."
                        )

                    if last_posted_available is None or abs(
                        available_spots - last_posted_available
                    ) >= MIN_CHANGE_TO_POST:
                        if post_update_to_backend(available_spots):
                            last_posted_available = available_spots

                if now - last_display_time >= min_display_dt:
                    last_display_time = now
                    disp = frame.copy()
                    if PREVIEW_BOX_OVERLAY_LIVE and last_yolo_result is not None:
                        nmap = getattr(last_yolo_result, "names", None) or {}
                        disp = draw_counted_vehicle_boxes(disp, last_yolo_result, nmap)
                    draw_preview_overlay(
                        disp,
                        last_vehicles,
                        last_available,
                        FRAME_INTERVAL_SECONDS,
                        just_ran_infer,
                    )
                    cv2.imshow("wyze_detector (Ctrl+C to quit)", disp)
                    cv2.waitKey(1)
                continue

            # No live window: read + infer only on the interval (saves CPU).
            if now - last_infer_time < FRAME_INTERVAL_SECONDS:
                time.sleep(0.05)
                continue

            last_infer_time = now

            ret, frame = read_frame(cap)
            if not ret or frame is None:
                print("[WARN] Failed to read frame from camera.")
                time.sleep(1.0)
                continue

            detected_cars, preview, _yolo_res = count_cars_in_frame(model, frame)
            available_spots = compute_available_spots(detected_cars)
            last_vehicles = detected_cars
            last_available = available_spots

            print(
                f"[INFO] Detected vehicles: {detected_cars}, "
                f"computed availableSpots: {available_spots}"
            )

            now2 = time.time()
            if (
                detected_cars == 0
                and WYZE_PROBE_WHEN_EMPTY
                and (now2 - last_probe_time) >= WYZE_PROBE_INTERVAL_S
            ):
                last_probe_time = now2
                _, bright_ok = frame_brightness_ok(frame)
                print(
                    "[PROBE] No vehicle boxes this scan. "
                    f"Frame OK={bright_ok}. "
                    f"Full-COCO hints (conf>={WYZE_PROBE_CONF}):"
                )
                for line in probe_all_coco_labels(model, frame):
                    print(line)
                print(
                    "[HINT] Vehicle pass uses YOLO_CONF + optional class filter. "
                    "Try $env:YOLO_CONF=\"0.15\"; $env:YOLO_VEHICLE_ONLY=\"0\" for debugging."
                )

            if last_posted_available is None or abs(
                available_spots - last_posted_available
            ) >= MIN_CHANGE_TO_POST:
                if post_update_to_backend(available_spots):
                    last_posted_available = available_spots

            if SHOW_PREVIEW and preview is not None:
                cv2.imshow("wyze_detector (Ctrl+C to quit)", preview)
                cv2.waitKey(1)

    except KeyboardInterrupt:
        print("\n[INFO] Stopping (Ctrl+C pressed).")
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

