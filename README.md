# VisionGuide AI: Real-Time Indoor Navigation & Spatial Perception Assistant

## 1. Problem Statement
Visually impaired people find it hard to move safely indoors. Obstacles, people and doors are hard to notice without sight.

## 2. Objective
Build a real-time system that uses a camera to detect objects, estimate relative depth, check which path (left / center / right) is free, decide a direction and speak it aloud.

## 3. Features
Live webcam input, YOLOv8 detection, MiDaS relative depth, YOLO+depth fusion, 3-zone path analysis, navigation decision, decision stabilization (no flickering voice), pyttsx3 voice, OpenCV live display, FastAPI backend, HTML dashboard with browser/PC camera, tests.

## 4. Architecture
```
Camera -> YOLOv8 -> MiDaS -> Fusion -> Spatial Analysis (L/C/R) -> Decision
       -> Stabilizer -> Voice -> Display
Python vision engine (engine.py) -> FastAPI (backend/main.py) -> HTML dashboard (index.html)
```
| File | Job |
|---|---|
| `config.py` | all settings (camera index, thresholds, image size) |
| `camera.py` / `camera_test.py` | webcam + test |
| `detector.py` | YOLOv8 detection |
| `depth.py` | MiDaS depth + fusion (`attach_depth`) |
| `spatial.py` | LEFT/CENTER/RIGHT analysis |
| `navigation.py` | decision + `DecisionStabilizer` |
| `speech.py` | voice (own thread) |
| `engine.py` | full pipeline |
| `markers.py` / `generate_markers.py` | ArUco marker detection / printable markers |
| `route.py` / `map.json` | map graph, shortest route, RouteNavigator |
| `main.py` | OpenCV app |
| `backend/main.py` | FastAPI |
| `index.html` | dashboard (single HTML file, served by the backend) |

## 5. Technologies
Python, OpenCV, YOLOv8 (Ultralytics), MiDaS, PyTorch, Torchvision, NumPy, Pillow, timm, pyttsx3, FastAPI, React (Vite).

## 6. Hardware
Ryzen 5 / Intel i5 or higher, 8 GB RAM minimum, built-in or USB webcam, speaker/headphones. No NVIDIA GPU needed.

## 7. Software
Windows 10/11 64-bit, Python 3.10-3.12, no Node.js needed.

## 8. Installation (PowerShell)
```powershell
New-Item -ItemType Directory -Force -Path "D:\Blind Assitence DL\VisionGuide-AI"
Set-Location "D:\Blind Assitence DL\VisionGuide-AI"
# copy the project files here, then:
python -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install torch torchvision
python -m pip install -r requirements.txt
python -c "import cv2, torch, ultralytics, timm, pyttsx3, fastapi; print('ALL IMPORTS OK')"
```
If scripts are blocked: `Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned`.
First run downloads `yolov8n.pt` (HuggingFace `Ultralytics/YOLOv8`) and the MiDaS weights (HuggingFace `Intel/dpt-swinv2-tiny-256`); internet needed once. Move `yolov8n.pt` into `models/` if you like.

> Use `python -m pip ...` (not `pip ...`) and `python -m pytest`, `python -m uvicorn`. This avoids the Windows "Application Control policy blocked pip.exe" error. On Windows the normal PyPI PyTorch is already CPU-only.

## 9. How to run
```powershell
python camera_test.py --scan          # find camera index
python camera_test.py                 # webcam test
python main.py                        # full app, press Q to quit
python main.py --camera 1 --no-speech --imgsz 256 --depth-every 5   # faster
python main.py --no-depth             # YOLO only
```
Dashboard (ONE terminal, no Node.js / npm needed):
```powershell
# project root, venv active
python -m uvicorn backend.main:app --port 8000
# then open  http://localhost:8000   (Chrome or Edge)
```
In the dashboard choose the camera source:
- **Start browser camera** - the browser asks "Allow camera?"; click Allow. Frames go to the AI backend (`POST /analyze`) and boxes are drawn over the live video. Works with the laptop webcam or a USB webcam.
- **Use PC camera (server)** - the Python backend opens the webcam itself (OpenCV) and streams the annotated video.
Only one source can use the webcam at a time; switching sources releases the camera automatically. If the backend is not running the page still opens in DEMO mode (and the browser camera works as a plain preview).
Set `$env:VISIONGUIDE_AUTOSTART=1` to open the PC camera as soon as the backend starts. Do not run `main.py` at the same time.

Browser API check: http://localhost:8000/health, /status, /detections, /navigation, /docs

## 10. Camera configuration
Edit `CAMERA_INDEX` in `config.py` (0, 1, 2) or use `--camera 1`. For backend: `$env:VISIONGUIDE_CAMERA=1` before uvicorn.

## 11. YOLO
YOLOv8 finds objects in one pass and gives class, confidence and a box. We use `yolov8n.pt` (nano) because it is fastest on CPU.
**Important:** the pretrained model knows the 80 COCO classes (person, chair, bottle, ...). It does **NOT** detect `door`, `stairs`, `elevator_door`, `exit_sign`.
- **Phase 1 (now):** use COCO classes for testing.
- **Phase 2 (custom training):** collect/label images (Roboflow or LabelImg) for `door, stairs, elevator_door, exit_sign, person, obstacle`, create `data.yaml`, then:
```powershell
yolo detect train data=data.yaml model=yolov8n.pt epochs=50 imgsz=416 device=cpu
```
Copy `runs/detect/train/weights/best.pt` to `models/indoor_best.pt` and set `YOLO_MODEL` in `config.py`. CPU training is slow; Google Colab (free GPU) is easier.

## 12. MiDaS
MiDaS is a neural network that guesses depth from one normal image. We load it from **HuggingFace** (`Intel/dpt-swinv2-tiny-256`, MiDaS 3.1, via `transformers`); change `DEPTH_HF_MODEL` in `config.py` for `Intel/dpt-hybrid-midas` / `Intel/dpt-large` (more accurate, slower). If HuggingFace loading fails it automatically falls back to `MiDaS_small` from torch.hub (`DEPTH_BACKEND = "torchhub"` forces that). Weights are cached in `models/hf_cache/`.

## 13. Relative depth
MiDaS gives **relative** depth, not meters. We normalise to 0-1 where **0 = far, 1 = near**, using a slowly adapting range (temporal smoothing, `RANGE_ALPHA` in `depth.py`) so a flat wall is not stretched into noise and a far object is not made to look near. Never say "the chair is 2 meters away". Say "the chair is relatively close".

## 14. Spatial analysis
The frame is split into three equal vertical zones. An object belongs to a zone if at least 25% of its box width is inside it. Zone status from the closest object: CLEAR (< 0.55), BLOCKED (>= 0.55), HIGH RISK (>= 0.75). All thresholds are in `config.py`. If depth fails, box size is used as a rough fallback.

## 15. Navigation logic
Center CLEAR -> "Move forward"; else left CLEAR -> "Turn left"; else right CLEAR -> "Turn right"; else "Stop". `DecisionStabilizer` announces a decision only after it repeats `STABLE_FRAMES` (5) times. The voice speaks only when the stable decision changes.
**This is an assistive prototype. It does not guarantee collision avoidance.**

## 16. FastAPI
`GET /health`, `/status`, `/detections`, `/navigation`, `/video` (live MJPEG), `/state` (everything in one call), `POST /analyze` (browser camera frame), `POST /server-camera`. The vision engine runs in a background thread.

## 17. Dashboard
`index.html` is one HTML file served at `http://localhost:8000`. It polls `/state` every 0.5 s (PC camera mode) or sends camera frames to `/analyze` (browser camera mode) and shows status, instruction, L/C/R zones, objects with relative depth, live video and the indoor route. Camera errors (permission blocked, no camera, camera busy) are shown as friendly messages, not crashes. Camera on another device (phone) needs HTTPS.

## 18. Testing
```powershell
pytest tests -v
```
Camera, YOLO and MiDaS tests are skipped automatically if no camera / model / internet is available. Spatial, navigation, stabilizer, speech and API tests run anywhere.

## 19. Performance (Ryzen 5 7520U, CPU only)
Expect lower FPS than a GPU (roughly a few FPS to ~10 with depth on; not measured, so test on your laptop). To speed up: `--imgsz 256`, `--depth-every 5`, `--no-depth`, lower webcam resolution in `config.py`, close other apps.

## 20. Limitations
Relative (not metric) depth; COCO model lacks door/stairs/exit_sign until custom trained; stairs/drop-offs are not detected by obstacle logic; poor lighting reduces accuracy; one RGB camera; not safety-certified.

## 21. Future enhancements
Custom indoor model, stairs/drop-off detection, metric depth with calibration or depth camera, mobile/wearable version, offline multilingual voice, haptic feedback.

## Voice guidance
Voice says one sentence: obstacle name + position + action, e.g. **"Laptop ahead. Turn left"**, **"Chair on your left. Move forward"**, or **"Wait"** (for Stop). Only obstacles with relative depth >= `BLOCKED_THRESHOLD` are announced (max `SPEECH_MAX_OBJECTS`). Same sentence repeats every `SPEECH_REPEAT_SECONDS`. Change wording in `SPOKEN_TEXT`, `FRIENDLY_NAMES`, `ZONE_WORDS`. Test: `python speech_test.py`. If pyttsx3 fails, Windows PowerShell speech is used automatically.

## Level 3: Marker-based indoor navigation (guided routes)
**Idea:** print ArUco markers and stick them on walls at eye level at junctions/places (Entrance, Reception, Library...). When the camera sees a marker, the system knows *where you are*, finds the shortest route to your destination using `map.json`, and speaks the next instruction. Obstacle warnings always have priority over route instructions.

**Setup (do once):**
1. Edit `map.json` for YOUR corridor: `nodes` (marker id + place name) and `edges` (instruction text, `steps`, optional `reverse_instruction`). Write each instruction as heard when standing in front of that marker, **facing it**.
2. `python generate_markers.py` -> printable PNGs in `outputs/markers/`. Print about 15 cm x 15 cm, keep the white border, paste flat at eye level where the user will face them.
3. Run: `python main.py --dest "Library"` (or press keys 1-9 in the window to choose a place, C to clear).

**How it works:** Camera -> ArUco detect -> big enough marker (`MARKER_MIN_SIZE` px, i.e. close enough) = "you are here" -> Dijkstra shortest path -> instruction e.g. "Reception. Turn left and walk about 15 steps" -> "You have reached Library". If you wander to another marker, the route is recalculated from there.

**Dashboard:** choose the destination from the dropdown. API: `GET /map`, `GET /route`, `POST /destination` with `{"destination": "Library"}`.

**Honest limits:** works only in an area where markers are placed (not an unprepared mall); location is known only at markers (no tracking between them); steps are approximate; if a marker is missed the user must keep walking until the next one. Obstacle detection is still the same assistive prototype. Always combine with a white cane. Tune `MARKER_MIN_SIZE` (bigger = announce later/closer).

## Walls and doors
- **Walls:** YOLO has no "wall" class. Walls/large flat obstacles are detected from MiDaS depth: if a big part of a zone is "near" it is reported as **"Obstacle"**. Tune `WALL_DEPTH_THRESHOLD` and `WALL_AREA_FRACTION` in `config.py` (depth range is smoothed over time, so this is approximate). Needs MiDaS (not `--no-depth`).
- **Doors, stairs, exit signs:** NOT detected by `yolov8n.pt`. Train a custom model (Section 11) and set `YOLO_MODEL`. A door-only model loses COCO classes (chair, laptop), so either train on combined classes or run two models (future work).

## Troubleshooting
- `Could not open camera`: try index 1/2, close Zoom/Teams, allow desktop apps in Windows camera privacy.
- MiDaS load error: check internet (first run), `pip install transformers huggingface_hub timm`; if it persists try `pip install timm==0.6.12`. App continues with `--no-depth`.
- No voice: check Windows sound output; the app keeps running without TTS.
- Dashboard shows DEMO: backend is not running; start it on port 8000 and open http://localhost:8000.
- Browser camera blocked: click the lock/camera icon in the address bar -> Allow. Camera busy: close Zoom/Teams, or press Stop camera.

---
# Report Correction Checklist
(I could only see your instructions, not the report itself, so check each item against the actual report text.)
1. **Architecture:** the report says FastAPI + React but the appendix shows only OpenCV `cv2.imshow()`. Add `engine.py`, `backend/main.py`, `frontend/` code and the diagram "Python engine -> FastAPI -> React".
2. **Typo:** `import cv248` in the MiDaS appendix -> `import cv2`.
3. **`navigation_decision`:** make the definition and every call use the same signature: `navigation_decision(spatial)` returning exactly "Move forward"/"Turn left"/"Turn right"/"Stop".
4. **YOLO classes:** state that `yolov8n.pt` detects COCO classes only; door, stairs, elevator_door and exit_sign need a custom-trained model (Phase 2). Do not claim they are detected unless you trained it and report its metrics.
5. **Depth wording:** replace any "distance in meters" with "relative depth (0 = far, 1 = near)".
6. **Screenshots:** remove the career/education platform screenshots; add screenshots of the camera test, live window (L/C/R status + decision), depth map, FastAPI `/docs`, React dashboard, and passing tests.
7. **References:** remove references unrelated to VisionGuide; keep YOLOv8/Ultralytics, MiDaS (Ranftl et al.), PyTorch, OpenCV, FastAPI, React, pyttsx3.
8. **Hardware/performance:** say CPU-only on Ryzen 5 7520U with measured FPS from your own runs.
9. **Safety statement:** add that the system is an assistive prototype, no guarantee of collision avoidance.
10. **Modules/appendix:** make the appendix code match the final files in this project (including `config.py`, `DecisionStabilizer`, threaded speech).
