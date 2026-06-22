from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Sequence, cast

import numpy as np


class FaceEmbedder:
    """Generate and compare face embeddings using DeepFace."""

    def __init__(
        self,
        recognition_model: str = "ArcFace",
        detector_backend: str = "yolov8n",
        align: bool = True,
        enforce_detection: bool = True,
        normalization: str = "ArcFace",
        l2_normalize: bool = True,
    ) -> None:
        self.recognition_model = recognition_model
        self.detector_backend = detector_backend
        self.align = align
        self.enforce_detection = enforce_detection
        self.normalization = normalization
        self.l2_normalize = l2_normalize

    def __repr__(self) -> str:
        return (
            "FaceEmbedder("
            f"recognition_model={self.recognition_model!r}, "
            f"detector_backend={self.detector_backend!r})"
        )

    def get_embedding(self, image: Any) -> np.ndarray:
        """Return the first detected face embedding as a 1D numpy array."""
        result = self.represent(image, max_faces=1)[0]
        return np.asarray(result["embedding"], dtype=np.float32)

    def get_embeddings(self, images: Sequence[Any]) -> list[np.ndarray]:
        """Return one embedding per image, using the first detected face in each image."""
        results = cast(list[list[dict[str, Any]]], self.represent(list(images), max_faces=1))
        return [
            np.asarray(image_results[0]["embedding"], dtype=np.float32)
            for image_results in results
        ]

    def represent(
        self,
        image: Any,
        max_faces: int | None = None,
    ) -> list[dict[str, Any]] | list[list[dict[str, Any]]]:
        """Return DeepFace representation records, including face area metadata."""
        deepface = self._deepface()
        return deepface.represent(
            img_path=image,
            model_name=self.recognition_model,
            detector_backend=self.detector_backend,
            enforce_detection=self.enforce_detection,
            align=self.align,
            normalization=self.normalization,
            max_faces=max_faces,
            l2_normalize=self.l2_normalize,
        )

    @staticmethod
    def cosine_similarity(embedding_a: Sequence[float], embedding_b: Sequence[float]) -> float:
        """Return cosine similarity in [-1, 1], where higher means more similar."""
        a = np.asarray(embedding_a, dtype=np.float32)
        b = np.asarray(embedding_b, dtype=np.float32)

        denominator = np.linalg.norm(a) * np.linalg.norm(b)
        if denominator == 0:
            raise ValueError("Cannot compare zero-length embeddings.")

        return float(np.dot(a, b) / denominator)

    @classmethod
    def is_match(
        cls,
        embedding_a: Sequence[float],
        embedding_b: Sequence[float],
        threshold: float = 0.68,
    ) -> bool:
        """Return True when cosine similarity is at or above the threshold."""
        return cls.cosine_similarity(embedding_a, embedding_b) >= threshold

    @staticmethod
    def _deepface() -> Any:
        project_dir = Path(__file__).resolve().parent
        os.environ.setdefault("DEEPFACE_HOME", str(project_dir))
        os.environ.setdefault("MPLCONFIGDIR", str(project_dir / ".cache" / "matplotlib"))

        try:
            from deepface import DeepFace
        except ValueError as exc:
            if "requires tf-keras package" in str(exc):
                raise RuntimeError(
                    "DeepFace cannot start with the installed TensorFlow version. "
                    "Install tf-keras in this environment or downgrade TensorFlow."
                ) from exc
            raise

        return DeepFace


Embedder = FaceEmbedder
