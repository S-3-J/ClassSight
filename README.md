# Face Attendance

A Python face-recognition attendance app built with Flask, DeepFace, TensorFlow,
OpenCV, and SQLite.

The app supports:

- Student enrollment from one or more face images.
- Persistent storage of students, embeddings, attendance sessions, and records.
- Attendance from uploaded videos.
- Live browser-camera recognition with visual face boxes.
- Attendance status based on first-seen and last-seen timestamps.

## Project Structure

```text
.
├── app.py                  # Flask routes and web UI
├── attendance.py           # Video/live frame processing and recognition
├── database.py             # SQLite persistence layer
├── embedder.py             # DeepFace embedding wrapper
├── requirements.txt        # Python dependencies
├── templates/              # Flask templates
├── attendance.db           # Local SQLite database, generated at runtime
├── uploads/                # Uploaded videos, generated at runtime
└── .deepface/weights/      # DeepFace model weights, generated at runtime
```

## Setup

Create and activate a virtual environment:

```bash
python3 -m venv venv
source venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the app:

```bash
python app.py
```

Open:

```text
http://127.0.0.1:5000
```

The first enrollment or recognition run may download model weights into
`.deepface/weights/`.

## Enrollment

Go to:

```text
/enroll
```

Enter:

- Student ID
- Student name
- One or more face images

The app stores a DeepFace embedding for each image in SQLite. One image can work,
but multiple images with different lighting, angle, and distance will make
recognition more reliable.

## Uploaded Video Attendance

Go to:

```text
/attendance
```

Upload a video and set:

- Class duration minutes
- Required attendance minutes
- Recognition similarity threshold
- Detection interval seconds

The app samples the video every `N` seconds, detects faces, compares each face
embedding with enrolled embeddings, and updates attendance records.

For each matched student:

- `first_seen_at` is set on first match.
- `last_seen_at` is updated on later matches.
- `detections` is incremented.
- `best_similarity` is retained.

Status logic:

- `present`: detected span is at least the required minutes.
- `presence_detected`: student was detected, but not long enough.
- `absent`: student was never matched.

## Live Attendance

Go to:

```text
/live
```

Use:

- `Open Camera` to start browser camera preview.
- `Start Session` to begin recording attendance.
- `Finish Session` to close the session and view results.

Live overlay colors:

- Green box: recognized enrolled student.
- Orange box: face detected but not recognized above threshold.

The live route can detect multiple faces in one frame. Every recognized face in
that frame is recorded for attendance.

## Threshold Notes

The default recognition threshold is `0.55`.

Higher values are stricter and reduce false positives, but may miss the same
person when lighting, angle, camera distance, or image quality changes.

Lower values are more permissive and increase matches, but can increase false
positives if many similar faces are enrolled.

For the current ArcFace cosine-similarity setup, tune this value with real class
footage and enrollment images.

## Data Storage

The app uses SQLite at:

```text
attendance.db
```

Runtime/generated files are ignored by git:

- `attendance.db`
- `uploads/`
- `.deepface/`
- `.cache/`
- `__pycache__/`
- `venv/`

## Current Limitations

- Video processing is synchronous, so long videos can block the web request.
- Attendance duration is calculated from first matched timestamp to last matched
  timestamp, not from continuous face presence.
- The live browser view sends sampled frames to Flask; it is not optimized for
  large classrooms yet.
- Anti-spoofing is not enabled.
- There is no authentication/admin login yet.

## Useful Commands

Syntax check:

```bash
python -m py_compile app.py attendance.py database.py embedder.py
```

Inspect database:

```bash
sqlite3 attendance.db ".tables"
sqlite3 attendance.db "select * from students;"
sqlite3 attendance.db "select * from attendance_sessions;"
sqlite3 attendance.db "select * from attendance_records;"
```

