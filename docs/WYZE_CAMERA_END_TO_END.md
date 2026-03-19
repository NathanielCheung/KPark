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
- runs YOLO to detect `car`, `truck`, `bus`, and `motorcycle`,
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
$env:WYZE_RTSP_URL="rtsp://localhost:8554/hottpotty"

# Detection tuning (can be adjusted):
$env:YOLO_CONF="0.1"        # lower => more sensitive, more false positives
$env:YOLO_IMGSZ="1280"      # 640 or 1280 are good starting points
$env:DETECTION_DEBUG="1"    # print each detection label + confidence
$env:SHOW_PREVIEW="1"       # show window with bounding boxes

python wyze_beamish_detector.py
```

You should see output like:

```text
=== Wyze -> Beamish Munro Hall parking bridge ===
RTSP_URL          = rtsp://localhost:8554/hottpotty
PARKING_API_URL   = http://localhost:3001/api/parking
LOT_ID            = ontario-brock-lot
TOTAL_SPOTS       = 3
FRAME_INTERVAL_S  = 1.0
YOLO_CONF         = 0.1
YOLO_IMGSZ        = 1280
DETECTION_DEBUG   = True
SHOW_PREVIEW      = True
[INFO] Video stream opened. Press Ctrl+C to stop.
[INFO] Detected cars: 2, computed availableSpots: 1
[OK] Posted update to backend: {'lotId': 'ontario-brock-lot', 'availableSpots': 1} (status 200)
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
  TOTAL_SPOTS = int(os.environ.get("WYZE_TOTAL_SPOTS", "3"))
  ```

  Or override in PowerShell:

  ```powershell
  $env:WYZE_TOTAL_SPOTS="4"
  ```

- **How often to analyze a frame**  

  ```python
  FRAME_INTERVAL_SECONDS = float(os.environ.get("FRAME_INTERVAL_SECONDS", "1.0"))
  ```

  Or:

  ```powershell
  $env:FRAME_INTERVAL_SECONDS="1.0"
  ```

- **Change threshold before posting**  

  ```python
  MIN_CHANGE_TO_POST = int(os.environ.get("MIN_CHANGE_TO_POST", "1"))
  ```

  Only when `availableSpots` changes by at least this many spots does it send an update.

---

## Quick checklist

- [ ] `docker compose up` running in `wyze-bridge/`
- [ ] `npm start` running in `server/`
- [ ] `npm run dev` running in the project root with `.env` pointing to `http://localhost:3001`
- [ ] `python wyze_beamish_detector.py` running with `WYZE_RTSP_URL` set to your RTSP stream

If all four are running and you see `[OK] Posted update to backend` in Terminal 4, the **camera → map** pipeline is working end-to-end.

