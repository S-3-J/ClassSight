from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from flask import Flask, flash, jsonify, redirect, render_template, request, url_for
from werkzeug.utils import secure_filename

from attendance import DETECTION_LEVELS, AttendanceTracker, VideoAttendanceProcessor
from database import AttendanceDB
from embedder import FaceEmbedder


BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)


app = Flask(__name__)
app.config["SECRET_KEY"] = "dev-secret-change-before-production"
app.config["MAX_CONTENT_LENGTH"] = 512 * 1024 * 1024

db = AttendanceDB()
embedder = FaceEmbedder(enforce_detection=True)


@app.get("/")
def index():
    return render_template(
        "index.html",
        students=db.list_students(),
        sessions=db.list_sessions(),
    )


@app.route("/enroll", methods=["GET", "POST"])
def enroll():
    if request.method == "GET":
        return render_template("enroll.html", students=db.list_students())

    student_id = request.form.get("student_id", "").strip()
    name = request.form.get("name", "").strip()
    files = request.files.getlist("images")

    if not student_id or not name:
        flash("Student ID and name are required.", "error")
        return redirect(url_for("enroll"))

    if not files or all(not file.filename for file in files):
        flash("Upload at least one face image.", "error")
        return redirect(url_for("enroll"))

    db.upsert_student(student_id, name)

    saved = 0
    failures = []
    for file in files:
        if not file.filename:
            continue

        try:
            image = decode_image(file.read())
            embedding = embedder.get_embedding(image)
            db.add_embedding(
                student_id=student_id,
                embedding=embedding,
                recognition_model=embedder.recognition_model,
                detector_backend=embedder.detector_backend,
            )
            saved += 1
        except Exception as exc:
            failures.append(f"{file.filename}: {exc}")

    if saved:
        flash(f"Enrolled {saved} embedding(s) for {name}.", "success")

    for failure in failures[:3]:
        flash(failure, "error")

    return redirect(url_for("enroll"))


@app.route("/attendance", methods=["GET", "POST"])
def attendance():
    if request.method == "GET":
        return render_template("attendance.html", students=db.list_students())

    title = request.form.get("title", "Class").strip() or "Class"
    class_duration = float(request.form.get("class_duration_minutes", 40))
    required_minutes = float(request.form.get("required_minutes", 30))
    threshold = float(request.form.get("similarity_threshold", 0.55))
    interval = float(request.form.get("detection_interval_seconds", 5))
    detection_level = request.form.get("detection_level", "low")
    enable_cropping = request.form.get("enable_cropping") == "on"
    video = request.files.get("video")

    if video is None or not video.filename:
        flash("Upload a video file to process.", "error")
        return redirect(url_for("attendance"))

    if required_minutes > class_duration:
        flash("Required attendance minutes cannot exceed class duration.", "error")
        return redirect(url_for("attendance"))

    filename = secure_filename(video.filename)
    video_path = UPLOAD_DIR / filename
    video.save(video_path)

    session_id = db.create_session(
        title=title,
        class_duration_minutes=class_duration,
        required_minutes=required_minutes,
        similarity_threshold=threshold,
        detection_interval_seconds=interval,
    )

    detector_backend = DETECTION_LEVELS.get(detection_level, "yolov8n")
    processor = VideoAttendanceProcessor(
        db=db,
        embedder=FaceEmbedder(enforce_detection=False, detector_backend=detector_backend),
        detection_level=detection_level,
        enable_cropping=enable_cropping,
    )
    stats = processor.process_video(
        video_path=video_path,
        session_id=session_id,
        similarity_threshold=threshold,
        detection_interval_seconds=interval,
    )

    flash(
        "Processed video: "
        f"{stats['frames_sampled']} sampled frame(s), "
        f"{stats['detections']} matched detection(s).",
        "success",
    )
    return redirect(url_for("attendance_result", session_id=session_id))


@app.get("/live")
def live():
    return render_template("live.html", students=db.list_students())


@app.post("/api/live/start")
def api_live_start():
    payload = request.get_json(silent=True) or {}
    title = str(payload.get("title") or "Live Class")
    class_duration = float(payload.get("class_duration_minutes") or 40)
    required_minutes = float(payload.get("required_minutes") or 30)
    threshold = float(payload.get("similarity_threshold") or 0.55)
    interval = float(payload.get("detection_interval_seconds") or 2)
    detection_level = str(payload.get("detection_level") or "low")
    enable_cropping = bool(payload.get("enable_cropping", False))

    if required_minutes > class_duration:
        return jsonify({"error": "Required minutes cannot exceed class duration."}), 400

    session_id = db.create_session(
        title=title,
        class_duration_minutes=class_duration,
        required_minutes=required_minutes,
        similarity_threshold=threshold,
        detection_interval_seconds=interval,
    )
    return jsonify({
        "session_id": session_id,
        "detection_level": detection_level,
        "enable_cropping": enable_cropping,
    })


@app.post("/api/live/recognize")
def api_live_recognize():
    image_file = request.files.get("frame")
    if image_file is None:
        return jsonify({"error": "No frame uploaded."}), 400

    session_id = request.form.get("session_id", type=int)
    threshold = request.form.get("similarity_threshold", type=float) or 0.55
    record_attendance = request.form.get("record_attendance") == "true" and session_id
    detection_level = request.form.get("detection_level", "low")
    enable_cropping = request.form.get("enable_cropping") == "true"

    detector_backend = DETECTION_LEVELS.get(detection_level, "yolov8n")
    tracker = AttendanceTracker(
        db=db,
        embedder=FaceEmbedder(enforce_detection=True, detector_backend=detector_backend),
        session_id=session_id or 0,
        similarity_threshold=threshold,
        detection_level=detection_level,
        enable_cropping=enable_cropping,
    )

    try:
        image = decode_image(image_file.read())
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400

    detected_at = datetime_now()
    identifications = tracker.process_frame(image, detected_at, record_attendance)

    detections = [
        {
            "student_id": ident.student_id,
            "name": ident.name,
            "similarity": ident.similarity,
            "recognized": True,
            "facial_area": ident.facial_area,
        }
        for ident in identifications
    ]

    return jsonify(
        {
            "detections": detections
        }
    )


@app.post("/api/live/finish")
def api_live_finish():
    payload = request.get_json(silent=True) or {}
    session_id = int(payload.get("session_id") or 0)
    if not session_id:
        return jsonify({"error": "No active session."}), 400

    db.complete_session(session_id)
    return jsonify({"result_url": url_for("attendance_result", session_id=session_id)})


@app.get("/attendance/<int:session_id>")
def attendance_result(session_id: int):
    session = db.get_session(session_id)
    if session is None:
        flash("Attendance session not found.", "error")
        return redirect(url_for("index"))

    return render_template(
        "attendance_result.html",
        session=session,
        records=db.list_attendance(session_id),
    )


def decode_image(image_bytes: bytes) -> np.ndarray:
    data = np.frombuffer(image_bytes, dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Could not read image.")
    return image


def datetime_now():
    from datetime import datetime

    return datetime.now()


if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)
