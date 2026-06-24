from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from database import AttendanceDB
from embedder import FaceEmbedder


DETECTION_LEVELS = {
    "low": "yolov8n",
    "medium": "yolov8m",
    "high": "yolov8l",
}

CROP_GRID_SIZE = 3
CROP_OVERLAP_RATIO = 0.25
IOU_THRESHOLD = 0.3


@dataclass(frozen=True)
class KnownEmbedding:
    student_id: str
    name: str
    embedding: np.ndarray


@dataclass(frozen=True)
class Identification:
    student_id: str
    name: str
    similarity: float
    facial_area: dict[str, Any] | None = None


@dataclass
class FaceDetection:
    x: float
    y: float
    w: float
    h: float
    confidence: float = 1.0

    def to_facial_area(self) -> dict[str, Any]:
        return {"x": int(self.x), "y": int(self.y), "w": int(self.w), "h": int(self.h)}


def compute_crop_regions(
    frame_width: int,
    frame_height: int,
    grid_size: int = CROP_GRID_SIZE,
    overlap_ratio: float = CROP_OVERLAP_RATIO,
) -> list[tuple[int, int, int, int]]:
    step_x = frame_width / grid_size
    step_y = frame_height / grid_size
    overlap_x = step_x * overlap_ratio
    overlap_y = step_y * overlap_ratio

    regions = []
    for row in range(grid_size):
        for col in range(grid_size):
            x1 = max(0, int(col * step_x - overlap_x))
            y1 = max(0, int(row * step_y - overlap_y))
            x2 = min(frame_width, int((col + 1) * step_x + overlap_x))
            y2 = min(frame_height, int((row + 1) * step_y + overlap_y))
            regions.append((x1, y1, x2, y2))

    return regions


def _to_global_coords(
    crop_box: tuple[int, int, int, int], face: dict[str, Any]
) -> FaceDetection:
    x1, y1, x2, y2 = crop_box
    area = face.get("facial_area", {})
    fx = area.get("x", 0)
    fy = area.get("y", 0)
    fw = area.get("w", 0)
    fh = area.get("h", 0)
    return FaceDetection(
        x=fx + x1,
        y=fy + y1,
        w=fw,
        h=fh,
        confidence=face.get("confidence", 1.0),
    )


def _box_iou(a: FaceDetection, b: FaceDetection) -> float:
    x1 = max(a.x, b.x)
    y1 = max(a.y, b.y)
    x2 = min(a.x + a.w, b.x + b.w)
    y2 = min(a.y + a.h, b.y + b.h)
    if x2 <= x1 or y2 <= y1:
        return 0.0
    inter = (x2 - x1) * (y2 - y1)
    area_a = a.w * a.h
    area_b = b.w * b.h
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def _deduplicate_detections(
    detections: list[FaceDetection], iou_threshold: float = IOU_THRESHOLD
) -> list[FaceDetection]:
    if not detections:
        return []
    detections = sorted(detections, key=lambda d: d.confidence, reverse=True)
    kept: list[FaceDetection] = []
    for det in detections:
        is_dup = any(_box_iou(det, k) > iou_threshold for k in kept)
        if not is_dup:
            kept.append(det)
    return kept


def process_frame_with_crop_regions(
    frame: np.ndarray,
    embedder: FaceEmbedder,
    grid_size: int = CROP_GRID_SIZE,
    overlap_ratio: float = CROP_OVERLAP_RATIO,
) -> list[dict[str, Any]]:
    h, w = frame.shape[:2]
    regions = compute_crop_regions(w, h, grid_size, overlap_ratio)

    all_faces: list[dict[str, Any]] = []
    for x1, y1, x2, y2 in regions:
        crop = frame[y1:y2, x1:x2]
        try:
            faces = embedder.represent(crop, max_faces=None)
            for face in faces:
                global_det = _to_global_coords((x1, y1, x2, y2), face)
                face["facial_area"] = global_det.to_facial_area()
                all_faces.append(face)
        except Exception:
            continue

    if not all_faces:
        return []

    raw_detections = [
        FaceDetection(
            x=face["facial_area"]["x"],
            y=face["facial_area"]["y"],
            w=face["facial_area"]["w"],
            h=face["facial_area"]["h"],
            confidence=face.get("confidence", 1.0),
        )
        for face in all_faces
    ]
    unique_indices = []
    kept = _deduplicate_detections(raw_detections)
    kept_coords = {(d.x, d.y, d.w, d.h) for d in kept}
    for i, face in enumerate(all_faces):
        fa = face["facial_area"]
        if (fa["x"], fa["y"], fa["w"], fa["h"]) in kept_coords:
            unique_indices.append(i)
            kept_coords.discard((fa["x"], fa["y"], fa["w"], fa["h"]))

    return [all_faces[i] for i in unique_indices]


class AttendanceTracker:
    def __init__(
        self,
        db: AttendanceDB,
        embedder: FaceEmbedder,
        session_id: int,
        similarity_threshold: float,
        detection_level: str = "low",
        enable_cropping: bool = False,
    ) -> None:
        self.db = db
        self.embedder = embedder
        self.session_id = session_id
        self.similarity_threshold = similarity_threshold
        self.detection_level = detection_level
        self.enable_cropping = enable_cropping
        self.known_embeddings = self._load_known_embeddings()

    def process_frame(
        self,
        frame: np.ndarray,
        detected_at: datetime,
        record_attendance: bool = True,
    ) -> list[Identification]:
        if not self.known_embeddings:
            return []

        try:
            if self.enable_cropping:
                face_records = process_frame_with_crop_regions(frame, self.embedder)
            else:
                face_records = self.embedder.represent(frame)
        except Exception:
            return []

        detections = []
        for face_record in face_records:
            embedding = np.asarray(face_record["embedding"], dtype=np.float32)
            identification = self.identify(embedding)
            if identification is None:
                continue

            identification = Identification(
                student_id=identification.student_id,
                name=identification.name,
                similarity=identification.similarity,
                facial_area=face_record.get("facial_area"),
            )
            if record_attendance:
                self.db.record_detection(
                    session_id=self.session_id,
                    student_id=identification.student_id,
                    detected_at=detected_at,
                    similarity=identification.similarity,
                )
            detections.append(identification)

        return detections

    def identify(self, embedding: np.ndarray) -> Identification | None:
        best_match = self.best_match(embedding)
        if best_match is None or best_match.similarity < self.similarity_threshold:
            return None

        return best_match

    def best_match(self, embedding: np.ndarray) -> Identification | None:
        best_match: Identification | None = None

        for known in self.known_embeddings:
            similarity = self.embedder.cosine_similarity(embedding, known.embedding)
            if best_match is None or similarity > best_match.similarity:
                best_match = Identification(
                    student_id=known.student_id,
                    name=known.name,
                    similarity=similarity,
                )

        return best_match

    def _load_known_embeddings(self) -> list[KnownEmbedding]:
        embeddings = []
        for row in self.db.iter_embeddings():
            embeddings.append(
                KnownEmbedding(
                    student_id=row["student_id"],
                    name=row["name"],
                    embedding=np.asarray(json.loads(row["embedding_json"]), dtype=np.float32),
                )
            )

        return embeddings


class VideoAttendanceProcessor:
    def __init__(
        self,
        db: AttendanceDB,
        embedder: FaceEmbedder | None = None,
        detection_level: str = "low",
        enable_cropping: bool = False,
    ) -> None:
        self.db = db
        self.embedder = embedder or FaceEmbedder(enforce_detection=False)
        self.detection_level = detection_level
        self.enable_cropping = enable_cropping

    def process_video(
        self,
        video_path: str | Path,
        session_id: int,
        similarity_threshold: float,
        detection_interval_seconds: float,
    ) -> dict[str, Any]:
        capture = cv2.VideoCapture(str(video_path))
        if not capture.isOpened():
            raise ValueError(f"Could not open video: {video_path}")

        tracker = AttendanceTracker(
            db=self.db,
            embedder=self.embedder,
            session_id=session_id,
            similarity_threshold=similarity_threshold,
            detection_level=self.detection_level,
            enable_cropping=self.enable_cropping,
        )

        fps = capture.get(cv2.CAP_PROP_FPS) or 0
        total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        duration_seconds = total_frames / fps if fps > 0 and total_frames > 0 else None

        start_time = datetime.now()
        next_sample_at = 0.0
        frames_seen = 0
        frames_sampled = 0
        detections = 0

        while True:
            ok, frame = capture.read()
            if not ok:
                break

            frames_seen += 1
            elapsed_seconds = self._elapsed_seconds(capture, fps, frames_seen)
            if elapsed_seconds + 1e-9 < next_sample_at:
                continue

            detected_at = start_time + timedelta(seconds=elapsed_seconds)
            frame_detections = tracker.process_frame(frame, detected_at)
            detections += len(frame_detections)
            frames_sampled += 1
            next_sample_at = elapsed_seconds + detection_interval_seconds

        capture.release()
        self.db.complete_session(session_id)

        return {
            "duration_seconds": duration_seconds,
            "frames_seen": frames_seen,
            "frames_sampled": frames_sampled,
            "detections": detections,
        }

    @staticmethod
    def _elapsed_seconds(capture: cv2.VideoCapture, fps: float, frames_seen: int) -> float:
        msec = capture.get(cv2.CAP_PROP_POS_MSEC)
        if msec and msec > 0:
            return msec / 1000

        if fps > 0:
            return frames_seen / fps

        return float(frames_seen)
