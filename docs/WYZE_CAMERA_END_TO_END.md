# Wyze camera → website: end-to-end setup

This file summarizes **everything you need running** for the Wyze camera to update the website with live parking availability.

There are **4 pieces**, usually in **4 separate terminals**:

1. Wyze Bridge (camera → RTSP)
2. Backend API (`server/`)
3. Frontend website (Vite app)
4. Wyze detector script (`wyze_beamish_detector.py`)

Assumptions:

- You are in **PowerShell** on Windows.
- Your project root is `QHacks_Meter`.
- Your Wyze Bridge shows an RTSP URL like `rtsp://localhost:8554/hottpotty`.

---

## Terminal 1 – Wyze Bridge (camera → RTSP)

From the project root:

```powershell
cd wyze-bridge
docker compose up
```

- Leave this running.
- Visit `http://localhost:5000` in your browser.
- Confirm your camera appears (e.g. name `hottpotty`).
- Under **Streams → RTSP**, you should see something like:

  ```text
  rtsp://localhost:8554/hottpotty
  ```

You will use that RTSP URL in Terminal 4.

---

## Terminal 2 – Backend API (`server/`)

From the project root:

```powershell
cd server
npm install      # first time only
npm start
```

You should see output indicating the API is listening on `http://localhost:3001`.

---

## Terminal 3 – Frontend website

From the project root:

1. Make sure you have a `.env` file with:

   ```env
   VITE_PARKING_API_URL=http://localhost:3001
   ```

2. Start the dev server:

   ```powershell
   npm install      # first time only
   npm run dev
   ```

3. Open the URL shown in the terminal (usually `http://localhost:5173`).

The website will poll the backend every few seconds for parking data.

---

## Terminal 4 – Wyze detector (RTSP → backend)

This script:

- reads RTSP from Wyze Bridge,
- runs YOLO and counts/draws only `car`, `truck`, and `motorcycle` (no bus/plane/person/etc. boxes),
- computes `availableSpots`,
- POSTs to the backend at `/api/parking`.

From the project root:

### One-time Python dependencies

You already did this once, but for completeness:

```powershell
pip install ultralytics opencv-python requests
```

### Run the detector

Replace the RTSP URL below with the exact one from Wyze Bridge if different:

```powershell
# From your QHacks_Meter project root (folder that contains wyze_beamish_detector.py)

$env:WYZE_RTSP_URL="rtsp://localhost:8554/hottpotty"   # match Wyze Bridge → RTSP

$env:WYZE_TOTAL_SPOTS="10"
# $env:WYZE_LOT_ID="ontario-brock-lot"   # optional; default is this

$env:FRAME_INTERVAL_SECONDS="2"

$env:YOLO_CONF="0.2"                    # lower (e.g. 0.15) if cars are missed
$env:YOLO_VEHICLE_ONLY="1"                # only ask model for car / motorcycle / truck
$env:YOLO_IMGSZ="1280"
$env:DETECTION_DEBUG="1"
$env:SHOW_PREVIEW="1"

$env:PREVIEW_LIVE="1"                     # live feed
$env:PREVIEW_BOX_OVERLAY_LIVE="1"        # car/truck/motorcycle boxes on live feed (default on)
$env:PREVIEW_MAX_FPS="20"

# Optional: turn off extra debug probe prints
# $env:WYZE_PROBE_WHEN_EMPTY="0"

python wyze_beamish_detector.py
```

**How availability works:** on each **scan** (every `FRAME_INTERVAL_SECONDS`), YOLO counts only **`car`, `truck`, and `motorcycle`**. The preview draws boxes for those classes only (not people, boats, clocks, etc.). With **`PREVIEW_LIVE=1`**, **`PREVIEW_BOX_OVERLAY_LIVE=1`** (default) redraws those outlines on the live feed using the last scan.  
`availableSpots = WYZE_TOTAL_SPOTS − that count` (never below 0). So **each extra vehicle in view reduces available spots by 1** until you hit zero.

You should see output like:

```text
=== Wyze -> Ontario & Brock Lot parking bridge ===
RTSP_URL          = rtsp://localhost:8554/hottpotty
PARKING_API_URL   = http://localhost:3001/api/parking
LOT_ID            = ontario-brock-lot
TOTAL_SPOTS       = 10
FRAME_INTERVAL_S  = 2.0
YOLO_CONF         = 0.4
YOLO_IMGSZ        = 1280
DETECTION_DEBUG   = True
SHOW_PREVIEW      = True
[INFO] Video stream opened. Press Ctrl+C to stop.
[INFO] Detected vehicles: 2, computed availableSpots: 8
[OK] Posted update to backend: {'lotId': 'ontario-brock-lot', 'availableSpots': 8} (status 200)
```

This means:

- YOLO is seeing vehicles in the frame.
- The script is POSTing updates to the backend.
- The website (Terminal 3) should update the lot labeled `Ontario & Brock Lot` accordingly.

---

## Tweaks you might want

- **How many spots the camera covers**  
  In `wyze_beamish_detector.py`:

  ```python
  TOTAL_SPOTS = int(os.environ.get("WYZE_TOTAL_SPOTS", "10"))
  ```

  Or override in PowerShell:

  ```powershell
  $env:WYZE_TOTAL_SPOTS="10"
  ```

- **How often to run YOLO / POST**  

  ```python
  FRAME_INTERVAL_SECONDS = float(os.environ.get("FRAME_INTERVAL_SECONDS", "2.0"))
  ```

  Or:

  ```powershell
  $env:FRAME_INTERVAL_SECONDS="2"
  ```

- **Preview window**  
  With `PREVIEW_LIVE=1` (default when `SHOW_PREVIEW=1`), the window shows a **live RTSP feed** and green text with the **last** vehicle count and availability; the **model still runs only every `FRAME_INTERVAL_SECONDS`**.  
  Bounding boxes from YOLO appear only if you set `PREVIEW_LIVE=0` (window updates once per scan).

- **Change threshold before posting**  

  ```python
  MIN_CHANGE_TO_POST = int(os.environ.get("MIN_CHANGE_TO_POST", "1"))
  ```

  Only when `availableSpots` changes by at least this many spots does it send an update.

- **Cars visible but count stays 0**  
  On Wyze RTSP, real cars often score **below 0.35–0.4** confidence, so Ultralytics never returns a box. Try:

  ```powershell
  $env:YOLO_CONF="0.2"          # or 0.15 if still missing
  $env:YOLO_IMGSZ="1280"
  $env:DETECTION_DEBUG="1"     # see every vehicle-class box and score
  ```

  Optional: use a slightly larger model (better on small cars, slower):

  ```powershell
  $env:YOLO_MODEL="yolov8s.pt"
  ```

  If you lowered `YOLO_CONF` and get too many phantom vehicles, add a **second** floor only for counting:

  ```powershell
  $env:YOLO_CONF="0.15"
  $env:VEHICLE_MIN_CONF="0.28"
  ```

  To see *all* COCO labels again (debug), turn off vehicle-only filtering:

  ```powershell
  $env:YOLO_VEHICLE_ONLY="0"
  ```

- **“Not detecting anything” / no `[DEBUG] detection` lines**  
  The script now defaults **RTSP over TCP on Windows**, **warms up** the stream, and prints a **`[PROBE]`** block every ~12s when the vehicle count is 0 (full COCO, low threshold). Read that output:
  - **Probe lists `car`/`truck`/`motorcycle` with `<-- counted vehicle`** → lower `$env:YOLO_CONF` (e.g. `0.15`) or set `$env:YOLO_VEHICLE_ONLY="0"` temporarily.
  - **Probe shows other labels only** → model is “seeing” the scene but not as a vehicle; aim the camera or try `$env:YOLO_MODEL="yolov8s.pt"`.
  - **Probe shows no boxes** → stream/frame problem: confirm preview window isn’t black; try `$env:WYZE_WARMUP_FRAMES="40"` or `$env:WYZE_PROBE_CONF="0.03"`.

- **Terminal: `[h264] Missing reference picture` / `decode_slice_header error`**  
  The H.264 decoder started mid-stream or dropped reference frames. The script now defaults to **TCP RTSP**, **FFmpeg `discardcorrupt`**, a **capture buffer of 4** (not 1), **longer warmup**, and a short pause between reads.  
  If errors persist: restart **Wyze Bridge** and the script; optionally try `$env:WYZE_CAP_BUFFERSIZE="6"` or `$env:WYZE_WARMUP_FRAMES="80"`.  
  To override FFmpeg options entirely (PowerShell, **before** `python`):

  ```powershell
  $env:OPENCV_FFMPEG_CAPTURE_OPTIONS="rtsp_transport;tcp|fflags;discardcorrupt"
  ```

---

## Quick checklist

- [ ] `docker compose up` running in `wyze-bridge/`
- [ ] `npm start` running in `server/`
- [ ] `npm run dev` running in the project root with `.env` pointing to `http://localhost:3001`
- [ ] `python wyze_beamish_detector.py` running with `WYZE_RTSP_URL` set to your RTSP stream

If all four are running and you see `[OK] Posted update to backend` in Terminal 4, the **camera → map** pipeline is working end-to-end.

