from __future__ import annotations

import json
import threading
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from .face_detector import FaceDetection
from .protected_store import ProtectedRepresentationStore
from .recognizer import SFaceRecognizer
from .student_store import StudentStore


# ============================================================
# RESULT MODEL
# ============================================================

@dataclass
class RecognitionResult:
    """One face-recognition decision for one frame."""

    track_id: int | None
    student_id: str | None
    name: str | None
    batch: str | None
    similarity: float
    status: str
    reason: str
    timestamp: str
    confirmation_count: int = 0

    def as_dict(self) -> dict[str, Any]:
        return {
            "track_id": self.track_id,
            "student_id": self.student_id,
            "name": self.name,
            "batch": self.batch,
            "similarity": round(float(self.similarity), 4),
            "confidence": round(float(self.similarity), 4),
            "status": self.status,
            "reason": self.reason,
            "timestamp": self.timestamp,
            "confirmation_count": self.confirmation_count,
        }


# ============================================================
# TRACK STATE
# ============================================================

@dataclass
class TrackIdentityState:
    """
    State for one tracked face.

    This is responsible for:
    - temporal confirmation
    - identity stability
    - temporary recognition failures
    - preventing rapid identity switching
    """

    candidate_id: str | None = None
    candidate_count: int = 0

    stable_id: str | None = None
    stable_similarity: float = 0.0

    switch_candidate_id: str | None = None
    switch_count: int = 0

    # Number of consecutive frames where recognition temporarily
    # failed after an identity was already confirmed.
    temporary_miss_count: int = 0

    last_seen: str | None = None

    recent_statuses: deque[str] = field(
        default_factory=lambda: deque(maxlen=10)
    )


# ============================================================
# RECOGNITION LOGGER
# ============================================================

class RecognitionEventLogger:
    """Append-only JSONL recognition log."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def write(self, event: dict[str, Any]) -> None:
        with self._lock:
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(
                    json.dumps(
                        event,
                        ensure_ascii=False,
                    )
                    + "\n"
                )

    def recent(self, limit: int = 50) -> list[dict[str, Any]]:
        limit = max(1, min(int(limit), 500))

        if not self.path.exists():
            return []

        with self._lock:
            lines = self.path.read_text(
                encoding="utf-8"
            ).splitlines()

        output: list[dict[str, Any]] = []

        for line in lines[-limit:]:
            try:
                output.append(json.loads(line))
            except json.JSONDecodeError:
                continue

        return output


# ============================================================
# FACE RECOGNITION SERVICE
# ============================================================

class FaceRecognitionService:
    """
    Phase 6 face recognition service.

    6.1  Face -> SFace representation
    6.2  Compare against enrolled students
    6.3  Configurable recognition threshold
    6.4  Identify recognized students
    6.5  Low-confidence -> unknown
    6.6  Per-track identity tracking
    6.7  Temporal confirmation
    6.8  Anti-switch stabilization
    6.9  Timestamp/confidence logging
    6.10 Known/unknown testing
    """

    def __init__(
        self,
        student_store: StudentStore,
        representation_store: ProtectedRepresentationStore,
        recognizer: SFaceRecognizer,
        log_path: str | Path,
        threshold: float = 0.45,
        confirmation_frames: int = 3,
        switch_confirmation_frames: int = 3,
        switch_margin: float = 0.05,
        max_track_idle_seconds: float = 3.0,

        # NEW:
        # Number of temporary failed frames allowed after an identity
        # has already been confirmed.
        recognition_grace_frames: int = 5,
    ):
        self.student_store = student_store
        self.representation_store = representation_store
        self.recognizer = recognizer

        self.logger = RecognitionEventLogger(log_path)

        self.threshold = float(threshold)

        self.confirmation_frames = max(
            1,
            int(confirmation_frames),
        )

        self.switch_confirmation_frames = max(
            1,
            int(switch_confirmation_frames),
        )

        self.switch_margin = max(
            0.0,
            float(switch_margin),
        )

        self.max_track_idle_seconds = max(
            0.5,
            float(max_track_idle_seconds),
        )

        self.recognition_grace_frames = max(
            0,
            int(recognition_grace_frames),
        )

        self._tracks: dict[int, TrackIdentityState] = {}

        self._lock = threading.RLock()

        self.last_results: list[RecognitionResult] = []

        self.total_recognitions = 0
        self.confirmed_count = 0
        self.unknown_count = 0

        self.last_inference_ms = 0.0
        self.last_error: str | None = None

    # ========================================================
    # HELPERS
    # ========================================================

    @staticmethod
    def _normalize(vector: np.ndarray) -> np.ndarray:
        array = np.asarray(
            vector,
            dtype=np.float32,
        ).reshape(-1)

        norm = float(np.linalg.norm(array))

        if norm <= 0:
            raise ValueError(
                "Face representation has zero norm."
            )

        return array / norm

    def _load_enrolled_students(self) -> list[dict[str, Any]]:
        students: list[dict[str, Any]] = []

        for public_student in self.student_store.list():
            full = self.student_store.get_with_representation(
                public_student["student_id"]
            )

            if not full:
                continue

            token = full.get("representation")

            if not token:
                continue

            try:
                vector = self.representation_store.reveal(
                    token
                )

                vector = self._normalize(vector)

            except Exception:
                continue

            full["_vector"] = vector

            students.append(full)

        return students

    @staticmethod
    def _cosine_similarity(
        a: np.ndarray,
        b: np.ndarray,
    ) -> float:
        a_norm = float(np.linalg.norm(a))
        b_norm = float(np.linalg.norm(b))

        if a_norm <= 0 or b_norm <= 0:
            return -1.0

        return float(
            np.dot(a, b)
            / (a_norm * b_norm)
        )

    @staticmethod
    def _now_iso() -> str:
        return datetime.now(
            timezone.utc
        ).isoformat()

    def _cleanup_idle_tracks(
        self,
        now: datetime,
    ) -> None:
        stale: list[int] = []

        for track_id, state in self._tracks.items():

            if not state.last_seen:
                continue

            try:
                last = datetime.fromisoformat(
                    state.last_seen
                )
            except ValueError:
                stale.append(track_id)
                continue

            if (
                now - last
            ).total_seconds() > self.max_track_idle_seconds:
                stale.append(track_id)

        for track_id in stale:
            self._tracks.pop(
                track_id,
                None,
            )

    def _match(
        self,
        feature: np.ndarray,
        students: list[dict[str, Any]],
    ) -> tuple[
        dict[str, Any] | None,
        float,
    ]:
        best_student = None
        best_similarity = -1.0

        for student in students:

            similarity = self._cosine_similarity(
                feature,
                student["_vector"],
            )

            if similarity > best_similarity:
                best_similarity = similarity
                best_student = student

        return (
            best_student,
            best_similarity,
        )

    def _find_student(
        self,
        student_id: str | None,
    ) -> dict[str, Any] | None:

        if not student_id:
            return None

        return self.student_store.get_with_representation(
            student_id
        )

    def _result(
        self,
        detection: FaceDetection,
        student: dict[str, Any] | None,
        similarity: float,
        status: str,
        reason: str,
        timestamp: str,
        confirmation_count: int = 0,
    ) -> RecognitionResult:

        return RecognitionResult(
            track_id=detection.track_id,
            student_id=(
                student.get("student_id")
                if student
                else None
            ),
            name=(
                student.get("name")
                if student
                else None
            ),
            batch=(
                student.get("batch")
                if student
                else None
            ),
            similarity=max(
                -1.0,
                float(similarity),
            ),
            status=status,
            reason=reason,
            timestamp=timestamp,
            confirmation_count=confirmation_count,
        )

    # ========================================================
    # TEMPORAL + STABILITY LOGIC
    # ========================================================

    def _apply_temporal_confirmation(
        self,
        detection: FaceDetection,
        candidate: dict[str, Any] | None,
        similarity: float,
        timestamp: str,
        failure_reason: str = "below_threshold",
    ) -> RecognitionResult:

        track_id = detection.track_id

        # ----------------------------------------------------
        # No tracking ID
        # ----------------------------------------------------

        if track_id is None:

            if candidate is None:
                return self._result(
                    detection,
                    None,
                    similarity,
                    "unknown",
                    failure_reason,
                    timestamp,
                )

            return self._result(
                detection,
                candidate,
                similarity,
                "confirmed",
                "no_track_id",
                timestamp,
                self.confirmation_frames,
            )

        # ----------------------------------------------------
        # Get/create state
        # ----------------------------------------------------

        state = self._tracks.setdefault(
            track_id,
            TrackIdentityState(),
        )

        state.last_seen = timestamp

        # ====================================================
        # CASE 1:
        # Recognition temporarily failed
        # ====================================================

        if candidate is None:

            # ------------------------------------------------
            # No previously confirmed identity
            # ------------------------------------------------

            if state.stable_id is None:

                state.candidate_id = None
                state.candidate_count = 0
                state.switch_candidate_id = None
                state.switch_count = 0

                state.recent_statuses.append(
                    "unknown"
                )

                return self._result(
                    detection,
                    None,
                    similarity,
                    "unknown",
                    failure_reason,
                    timestamp,
                )

            # ------------------------------------------------
            # IMPORTANT FIX:
            #
            # We already know this face.
            #
            # A single bad frame must NOT immediately
            # turn the student into Unknown.
            # ------------------------------------------------

            state.temporary_miss_count += 1

            if (
                state.temporary_miss_count
                <= self.recognition_grace_frames
            ):

                stable_student = self._find_student(
                    state.stable_id
                )

                state.recent_statuses.append(
                    "temporary_miss"
                )

                return self._result(
                    detection,
                    stable_student,
                    state.stable_similarity,
                    "confirmed",
                    (
                        "temporary_recognition_failure_"
                        f"held_identity_"
                        f"{state.temporary_miss_count}_of_"
                        f"{self.recognition_grace_frames}"
                    ),
                    timestamp,
                    self.confirmation_frames,
                )

            # ------------------------------------------------
            # Grace window exceeded
            # ------------------------------------------------

            stable_student = self._find_student(
                state.stable_id
            )

            state.stable_id = None
            state.stable_similarity = 0.0

            state.candidate_id = None
            state.candidate_count = 0

            state.switch_candidate_id = None
            state.switch_count = 0

            state.temporary_miss_count = 0

            state.recent_statuses.append(
                "unknown"
            )

            return self._result(
                detection,
                None,
                similarity,
                "unknown",
                (
                    "recognition_grace_exceeded_"
                    f"{self.recognition_grace_frames}_frames"
                ),
                timestamp,
            )

        # ====================================================
        # CASE 2:
        # We have a valid candidate
        # ====================================================

        candidate_id = candidate["student_id"]

        # Successful recognition resets temporary failures.

        state.temporary_miss_count = 0

        # ====================================================
        # No stable identity yet
        # ====================================================

        if state.stable_id is None:

            if state.candidate_id == candidate_id:
                state.candidate_count += 1

            else:
                state.candidate_id = candidate_id
                state.candidate_count = 1

            state.stable_similarity = similarity

            if (
                state.candidate_count
                >= self.confirmation_frames
            ):

                state.stable_id = candidate_id

                state.switch_candidate_id = None
                state.switch_count = 0

                state.recent_statuses.append(
                    "confirmed"
                )

                return self._result(
                    detection,
                    candidate,
                    similarity,
                    "confirmed",
                    "temporal_confirmation_complete",
                    timestamp,
                    state.candidate_count,
                )

            state.recent_statuses.append(
                "candidate"
            )

            return self._result(
                detection,
                candidate,
                similarity,
                "candidate",
                "awaiting_temporal_confirmation",
                timestamp,
                state.candidate_count,
            )

        # ====================================================
        # CASE 3:
        # Same identity as currently confirmed
        # ====================================================

        if candidate_id == state.stable_id:

            state.candidate_id = candidate_id

            state.candidate_count = (
                self.confirmation_frames
            )

            state.switch_candidate_id = None
            state.switch_count = 0

            state.stable_similarity = similarity

            state.recent_statuses.append(
                "confirmed"
            )

            return self._result(
                detection,
                candidate,
                similarity,
                "confirmed",
                "stable_identity",
                timestamp,
                state.candidate_count,
            )

        # ====================================================
        # CASE 4:
        # Different identity detected
        #
        # DO NOT switch immediately.
        # ====================================================

        stable_student = self._find_student(
            state.stable_id
        )

        stable_similarity = (
            state.stable_similarity
        )

        # ----------------------------------------------------
        # Candidate does not beat current identity enough
        # ----------------------------------------------------

        if (
            similarity
            < stable_similarity
            + self.switch_margin
        ):

            state.switch_candidate_id = None
            state.switch_count = 0

            state.recent_statuses.append(
                "confirmed"
            )

            return self._result(
                detection,
                stable_student or candidate,
                stable_similarity,
                "confirmed",
                "switch_rejected_by_margin",
                timestamp,
                self.confirmation_frames,
            )

        # ----------------------------------------------------
        # Candidate is strong enough to potentially switch
        # ----------------------------------------------------

        if (
            state.switch_candidate_id
            == candidate_id
        ):

            state.switch_count += 1

        else:

            state.switch_candidate_id = candidate_id
            state.switch_count = 1

        # ----------------------------------------------------
        # Confirm identity switch
        # ----------------------------------------------------

        if (
            state.switch_count
            >= self.switch_confirmation_frames
        ):

            state.stable_id = candidate_id

            state.stable_similarity = similarity

            state.switch_candidate_id = None
            state.switch_count = 0

            state.candidate_id = candidate_id
            state.candidate_count = (
                self.confirmation_frames
            )

            state.temporary_miss_count = 0

            state.recent_statuses.append(
                "confirmed"
            )

            return self._result(
                detection,
                candidate,
                similarity,
                "confirmed",
                "identity_switch_confirmed",
                timestamp,
                state.candidate_count,
            )

        # ----------------------------------------------------
        # Hold previous identity while waiting
        # ----------------------------------------------------

        state.recent_statuses.append(
            "confirmed"
        )

        return self._result(
            detection,
            stable_student or candidate,
            (
                state.stable_similarity
                if stable_student
                else similarity
            ),
            "confirmed",
            "holding_previous_identity",
            timestamp,
            self.confirmation_frames,
        )

    # ========================================================
    # MAIN RECOGNITION
    # ========================================================

    def recognize(
        self,
        frame: np.ndarray,
        detections: list[FaceDetection],
        *,
        camera_id: str | None = None,
        frame_id: str | None = None,
        timestamp: str | None = None,
    ) -> list[RecognitionResult]:

        started = datetime.now(
            timezone.utc
        )

        timestamp = (
            timestamp
            or self._now_iso()
        )

        results: list[RecognitionResult] = []

        try:

            # ------------------------------------------------
            # Load enrolled representations
            # ------------------------------------------------

            students = self._load_enrolled_students()

            with self._lock:

                self._cleanup_idle_tracks(
                    started
                )

                active_track_ids = {
                    d.track_id
                    for d in detections
                    if d.track_id is not None
                }

                # --------------------------------------------
                # Remove truly stale tracks
                # --------------------------------------------

                for stale_id in list(
                    self._tracks
                ):

                    if stale_id in active_track_ids:
                        continue

                    state = self._tracks[
                        stale_id
                    ]

                    if not state.last_seen:
                        continue

                    try:

                        last = datetime.fromisoformat(
                            state.last_seen
                        )

                        if (
                            started - last
                        ).total_seconds() > (
                            self.max_track_idle_seconds
                        ):

                            self._tracks.pop(
                                stale_id,
                                None,
                            )

                    except ValueError:

                        self._tracks.pop(
                            stale_id,
                            None,
                        )

                # --------------------------------------------
                # Process each detected face
                # --------------------------------------------

                for detection in detections:

                    # ========================================
                    # LOW QUALITY
                    # ========================================

                    if not detection.quality_valid:

                        # IMPORTANT:
                        # Do NOT directly return unknown.
                        # Let temporal logic decide whether
                        # an already-confirmed identity should
                        # be temporarily held.

                        result = (
                            self._apply_temporal_confirmation(
                                detection,
                                None,
                                0.0,
                                timestamp,
                                failure_reason="low_quality",
                            )
                        )

                    # ========================================
                    # NO ENROLLED STUDENTS
                    # ========================================

                    elif not students:

                        result = (
                            self._apply_temporal_confirmation(
                                detection,
                                None,
                                0.0,
                                timestamp,
                                failure_reason=(
                                    "no_enrolled_students"
                                ),
                            )
                        )

                    # ========================================
                    # YUNET LANDMARKS MISSING
                    # ========================================

                    elif detection.model_row is None:

                        result = (
                            self._apply_temporal_confirmation(
                                detection,
                                None,
                                0.0,
                                timestamp,
                                failure_reason=(
                                    "missing_yunet_landmarks"
                                ),
                            )
                        )

                    # ========================================
                    # NORMAL RECOGNITION
                    # ========================================

                    else:

                        feature = (
                            self.recognizer.extract(
                                frame,
                                detection.model_row,
                            )
                        )

                        candidate, similarity = (
                            self._match(
                                feature,
                                students,
                            )
                        )

                        # ------------------------------------
                        # Below threshold
                        # ------------------------------------

                        if (
                            candidate is None
                            or similarity
                            < self.threshold
                        ):

                            result = (
                                self._apply_temporal_confirmation(
                                    detection,
                                    None,
                                    similarity,
                                    timestamp,
                                    failure_reason=(
                                        "below_threshold"
                                    ),
                                )
                            )

                        # ------------------------------------
                        # Valid candidate
                        # ------------------------------------

                        else:

                            result = (
                                self._apply_temporal_confirmation(
                                    detection,
                                    candidate,
                                    similarity,
                                    timestamp,
                                )
                            )

                    # ========================================
                    # SAVE RESULT
                    # ========================================

                    results.append(result)

                    self.total_recognitions += 1

                    if result.status == "confirmed":
                        self.confirmed_count += 1

                    elif result.status == "unknown":
                        self.unknown_count += 1

                    # ========================================
                    # LOG RESULT
                    # ========================================

                    self.logger.write(
                        {
                            "timestamp": result.timestamp,
                            "camera_id": camera_id,
                            "frame_id": frame_id,
                            "track_id": result.track_id,
                            "student_id": result.student_id,
                            "name": result.name,
                            "batch": result.batch,
                            "similarity": round(
                                float(
                                    result.similarity
                                ),
                                6,
                            ),
                            "confidence": round(
                                float(
                                    result.similarity
                                ),
                                6,
                            ),
                            "status": result.status,
                            "reason": result.reason,
                            "confirmation_count": (
                                result.confirmation_count
                            ),
                        }
                    )

                self.last_results = results

            self.last_error = None

        except Exception as exc:

            self.last_error = str(exc)

            raise

        finally:

            elapsed = (
                datetime.now(
                    timezone.utc
                )
                - started
            ).total_seconds() * 1000.0

            self.last_inference_ms = elapsed

        return results

    # ========================================================
    # STATUS
    # ========================================================

    def status(self) -> dict[str, Any]:

        with self._lock:

            return {
                "model": "SFace",

                "threshold": round(
                    self.threshold,
                    4,
                ),

                "confirmation_frames": (
                    self.confirmation_frames
                ),

                "switch_confirmation_frames": (
                    self.switch_confirmation_frames
                ),

                "switch_margin": round(
                    self.switch_margin,
                    4,
                ),

                "recognition_grace_frames": (
                    self.recognition_grace_frames
                ),

                "active_identity_tracks": (
                    len(self._tracks)
                ),

                "total_recognitions": (
                    self.total_recognitions
                ),

                "confirmed_count": (
                    self.confirmed_count
                ),

                "unknown_count": (
                    self.unknown_count
                ),

                "last_inference_ms": round(
                    self.last_inference_ms,
                    3,
                ),

                "last_results": [
                    result.as_dict()
                    for result in self.last_results
                ],

                "last_error": self.last_error,

                "log_path": str(
                    self.logger.path
                ),
            }

    # ========================================================
    # CONFIGURATION
    # ========================================================

    def configure(
        self,
        *,
        threshold: float | None = None,
        confirmation_frames: int | None = None,
        switch_confirmation_frames: int | None = None,
        switch_margin: float | None = None,
        recognition_grace_frames: int | None = None,
    ) -> dict[str, Any]:

        if threshold is not None:

            if not 0.0 <= float(threshold) <= 1.0:
                raise ValueError(
                    "Recognition threshold must be between 0 and 1."
                )

            self.threshold = float(
                threshold
            )

        if confirmation_frames is not None:

            self.confirmation_frames = max(
                1,
                int(confirmation_frames),
            )

        if switch_confirmation_frames is not None:

            self.switch_confirmation_frames = max(
                1,
                int(switch_confirmation_frames),
            )

        if switch_margin is not None:

            self.switch_margin = max(
                0.0,
                float(switch_margin),
            )

        if recognition_grace_frames is not None:

            self.recognition_grace_frames = max(
                0,
                int(recognition_grace_frames),
            )

        return self.status()

    # ========================================================
    # RESET
    # ========================================================

    def reset(self) -> None:

        with self._lock:

            self._tracks.clear()

            self.last_results = []

            self.total_recognitions = 0
            self.confirmed_count = 0
            self.unknown_count = 0

            self.last_inference_ms = 0.0
            self.last_error = None