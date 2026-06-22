from __future__ import annotations

import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np


DB_PATH = Path(__file__).resolve().parent / "attendance.db"


class AttendanceDB:
    def __init__(self, path: str | Path = DB_PATH) -> None:
        self.path = Path(path)
        self.init_db()

    def connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        return conn

    def init_db(self) -> None:
        with self.connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS students (
                    student_id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS face_embeddings (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    student_id TEXT NOT NULL,
                    embedding_json TEXT NOT NULL,
                    recognition_model TEXT NOT NULL,
                    detector_backend TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY (student_id) REFERENCES students(student_id)
                        ON DELETE CASCADE
                );

                CREATE TABLE IF NOT EXISTS attendance_sessions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    title TEXT NOT NULL,
                    class_duration_minutes REAL NOT NULL,
                    required_minutes REAL NOT NULL,
                    similarity_threshold REAL NOT NULL,
                    detection_interval_seconds REAL NOT NULL,
                    started_at TEXT NOT NULL,
                    completed_at TEXT
                );

                CREATE TABLE IF NOT EXISTS attendance_records (
                    session_id INTEGER NOT NULL,
                    student_id TEXT NOT NULL,
                    first_seen_at TEXT NOT NULL,
                    last_seen_at TEXT NOT NULL,
                    detections INTEGER NOT NULL DEFAULT 1,
                    best_similarity REAL NOT NULL,
                    FOREIGN KEY (session_id) REFERENCES attendance_sessions(id)
                        ON DELETE CASCADE,
                    FOREIGN KEY (student_id) REFERENCES students(student_id)
                        ON DELETE CASCADE,
                    PRIMARY KEY (session_id, student_id)
                );
                """
            )

    def upsert_student(self, student_id: str, name: str) -> None:
        now = datetime.now().isoformat(timespec="seconds")
        with self.connect() as conn:
            conn.execute(
                """
                INSERT INTO students (student_id, name, created_at)
                VALUES (?, ?, ?)
                ON CONFLICT(student_id) DO UPDATE SET name = excluded.name
                """,
                (student_id.strip(), name.strip(), now),
            )

    def add_embedding(
        self,
        student_id: str,
        embedding: Sequence[float],
        recognition_model: str,
        detector_backend: str,
    ) -> None:
        now = datetime.now().isoformat(timespec="seconds")
        embedding_json = json.dumps(np.asarray(embedding, dtype=float).tolist())
        with self.connect() as conn:
            conn.execute(
                """
                INSERT INTO face_embeddings (
                    student_id,
                    embedding_json,
                    recognition_model,
                    detector_backend,
                    created_at
                )
                VALUES (?, ?, ?, ?, ?)
                """,
                (student_id, embedding_json, recognition_model, detector_backend, now),
            )

    def iter_embeddings(self) -> Iterable[sqlite3.Row]:
        with self.connect() as conn:
            rows = conn.execute(
                """
                SELECT
                    e.id,
                    e.student_id,
                    s.name,
                    e.embedding_json,
                    e.recognition_model,
                    e.detector_backend
                FROM face_embeddings e
                JOIN students s ON s.student_id = e.student_id
                ORDER BY s.name, e.id
                """
            ).fetchall()
        return rows

    def list_students(self) -> list[sqlite3.Row]:
        with self.connect() as conn:
            return conn.execute(
                """
                SELECT
                    s.student_id,
                    s.name,
                    s.created_at,
                    COUNT(e.id) AS embedding_count
                FROM students s
                LEFT JOIN face_embeddings e ON e.student_id = s.student_id
                GROUP BY s.student_id
                ORDER BY s.name
                """
            ).fetchall()

    def create_session(
        self,
        title: str,
        class_duration_minutes: float,
        required_minutes: float,
        similarity_threshold: float,
        detection_interval_seconds: float,
    ) -> int:
        started_at = datetime.now().isoformat(timespec="seconds")
        with self.connect() as conn:
            cursor = conn.execute(
                """
                INSERT INTO attendance_sessions (
                    title,
                    class_duration_minutes,
                    required_minutes,
                    similarity_threshold,
                    detection_interval_seconds,
                    started_at
                )
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    title.strip(),
                    class_duration_minutes,
                    required_minutes,
                    similarity_threshold,
                    detection_interval_seconds,
                    started_at,
                ),
            )
            return int(cursor.lastrowid)

    def record_detection(
        self,
        session_id: int,
        student_id: str,
        detected_at: datetime,
        similarity: float,
    ) -> None:
        seen_at = detected_at.isoformat(timespec="seconds")
        with self.connect() as conn:
            conn.execute(
                """
                INSERT INTO attendance_records (
                    session_id,
                    student_id,
                    first_seen_at,
                    last_seen_at,
                    detections,
                    best_similarity
                )
                VALUES (?, ?, ?, ?, 1, ?)
                ON CONFLICT(session_id, student_id) DO UPDATE SET
                    last_seen_at = excluded.last_seen_at,
                    detections = attendance_records.detections + 1,
                    best_similarity = MAX(attendance_records.best_similarity, excluded.best_similarity)
                """,
                (session_id, student_id, seen_at, seen_at, similarity),
            )

    def complete_session(self, session_id: int) -> None:
        completed_at = datetime.now().isoformat(timespec="seconds")
        with self.connect() as conn:
            conn.execute(
                "UPDATE attendance_sessions SET completed_at = ? WHERE id = ?",
                (completed_at, session_id),
            )

    def get_session(self, session_id: int) -> sqlite3.Row | None:
        with self.connect() as conn:
            return conn.execute(
                "SELECT * FROM attendance_sessions WHERE id = ?",
                (session_id,),
            ).fetchone()

    def list_sessions(self) -> list[sqlite3.Row]:
        with self.connect() as conn:
            return conn.execute(
                "SELECT * FROM attendance_sessions ORDER BY started_at DESC"
            ).fetchall()

    def list_attendance(self, session_id: int) -> list[dict[str, object]]:
        session = self.get_session(session_id)
        if session is None:
            return []

        required_minutes = float(session["required_minutes"])
        with self.connect() as conn:
            rows = conn.execute(
                """
                SELECT
                    s.student_id,
                    s.name,
                    r.first_seen_at,
                    r.last_seen_at,
                    r.detections,
                    r.best_similarity
                FROM students s
                LEFT JOIN attendance_records r
                    ON r.student_id = s.student_id
                    AND r.session_id = ?
                ORDER BY s.name
                """,
                (session_id,),
            ).fetchall()

        attendance = []
        for row in rows:
            attended_minutes = 0.0
            status = "absent"
            if row["first_seen_at"] and row["last_seen_at"]:
                first_seen = datetime.fromisoformat(row["first_seen_at"])
                last_seen = datetime.fromisoformat(row["last_seen_at"])
                attended_minutes = max(
                    0.0,
                    (last_seen - first_seen).total_seconds() / 60,
                )
                status = (
                    "present"
                    if attended_minutes >= required_minutes
                    else "presence_detected"
                )

            attendance.append(
                {
                    "student_id": row["student_id"],
                    "name": row["name"],
                    "first_seen_at": row["first_seen_at"],
                    "last_seen_at": row["last_seen_at"],
                    "detections": row["detections"] or 0,
                    "best_similarity": row["best_similarity"],
                    "attended_minutes": attended_minutes,
                    "status": status,
                }
            )

        return attendance
