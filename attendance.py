from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from database import AttendanceDB
from embedder import FaceEmbedder


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


class AttendanceTracker:
    def __init__(
        self,
        db: AttendanceDB,
        embedder: FaceEmbedder,
        session_id: int,
        similarity_threshold: float,
    ) -> None:
        self.db = db
        self.embedder = embedder
        self.session_id = session_id
        self.similarity_threshold = similarity_threshold
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
    ) -> None:
        self.db = db
        self.embedder = embedder or FaceEmbedder(enforce_detection=False)

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
