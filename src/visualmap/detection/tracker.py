"""Geometric association with motion prediction and temporal class evidence."""

from __future__ import annotations

from dataclasses import dataclass, field
from math import isfinite

import numpy as np

from visualmap.detection.types import Detection, TrackedDetection
from visualmap.detection.yolo_onnx import COCO80_NAMES

__all__ = ["IoUTracker", "TrackerConfig"]


@dataclass
class TrackerConfig:
    iou_threshold: float = 0.3
    max_age: int = 10
    min_hits: int = 1
    open_vocabulary_min_hits: int = 1
    class_vote_window: int = 8
    class_switch_hits: int = 3
    class_switch_ratio: float = 1.2
    class_mismatch_iou: float = 0.65
    motion_prediction: bool = True
    ambiguity_groups: tuple[tuple[str, ...], ...] = (("bottle", "cup"), ("laptop", "tv"))
    duplicate_iou: float = 0.85

    def __post_init__(self) -> None:
        if not 0 < self.iou_threshold <= 1:
            raise ValueError("tracker iou_threshold must be in (0, 1]")
        if not 0 < self.class_mismatch_iou <= 1:
            raise ValueError("class_mismatch_iou must be in (0, 1]")
        if self.max_age < 0 or self.min_hits < 1:
            raise ValueError("tracker max_age must be nonnegative and min_hits positive")
        if (
            isinstance(self.open_vocabulary_min_hits, bool)
            or not isinstance(self.open_vocabulary_min_hits, int)
            or self.open_vocabulary_min_hits < 1
        ):
            raise ValueError("open_vocabulary_min_hits must be a positive integer")
        if self.class_vote_window < 1 or not 1 <= self.class_switch_hits <= self.class_vote_window:
            raise ValueError("class_switch_hits must be within the class_vote_window")
        if not isfinite(self.class_switch_ratio) or self.class_switch_ratio < 1:
            raise ValueError("class_switch_ratio must be finite and at least 1")
        if not 0 < self.duplicate_iou <= 1:
            raise ValueError("duplicate_iou must be in (0, 1]")
        if any(len(group) < 2 or len(set(group)) != len(group) for group in self.ambiguity_groups):
            raise ValueError("ambiguity_groups need at least two distinct class names each")


@dataclass
class _Track:
    track_id: int
    class_id: int
    class_name: str
    xyxy: tuple[float, float, float, float]
    confidence: float
    hits: int = 1
    age: int = 0
    time_since_update: int = 0
    first_frame: int = 0
    last_frame: int = 0
    history: list[Detection] = field(default_factory=list)
    velocity: tuple[float, float] = (0.0, 0.0)
    class_consistency: float = 1.0
    class_observations: int = 1
    class_alternatives: tuple[Detection, ...] = ()


class IoUTracker:
    """Associate boxes globally and stabilize labels using recent observations.

    Prediction uses image pixels per source frame, so sampled video frame IDs
    do not turn a short occlusion into an arbitrary jump. Cross-class association
    requires stronger geometric agreement than same-class association.
    """

    def __init__(self, config: TrackerConfig | None = None) -> None:
        self.config = config or TrackerConfig()
        self._tracks: list[_Track] = []
        self._next_id = 1

    def update(self, detections: list[Detection]) -> list[TrackedDetection]:
        # Invalid model output must not create tracks or poison future geometry.
        detections = [d for d in detections if _valid_detection(d)]
        detections, alternatives = self._suppress_ambiguous_duplicates(detections)
        if not detections and not self._tracks:
            return []

        weights = np.full((len(self._tracks), len(detections)), -1.0)
        for ti, trk in enumerate(self._tracks):
            for di, det in enumerate(detections):
                # A structure proposal and a model object remain independent
                # hypotheses, even when their current image boxes coincide.
                if _evidence_source(trk.history[-1]) != _evidence_source(det):
                    continue
                predicted = self._predicted_box(trk, det.frame_id)
                overlap = _iou(predicted, det.xyxy)
                same_class = trk.class_id == det.class_id
                threshold = self.config.iou_threshold
                if not same_class:
                    threshold = max(threshold, self.config.class_mismatch_iou)
                # A sudden stop can invalidate velocity while retaining a strong
                # observed-box match. Do not accept a weak fallback at crossings.
                if overlap < threshold:
                    observed_overlap = _iou(trk.xyxy, det.xyxy)
                    if observed_overlap >= max(threshold, 0.65):
                        overlap = observed_overlap * 0.9
                if overlap >= threshold:
                    # Prefer a class-consistent association when geometry is similar.
                    weights[ti, di] = overlap - (0.0 if same_class else 0.15)

        matches = _maximum_weight_matching(weights)
        matched_trk = {ti for ti, _ in matches}
        matched_det = {di for _, di in matches}
        for ti, di in matches:
            trk, det = self._tracks[ti], detections[di]
            dt = det.frame_id - trk.last_frame
            if dt > 0:
                old_cx = (trk.xyxy[0] + trk.xyxy[2]) / 2
                old_cy = (trk.xyxy[1] + trk.xyxy[3]) / 2
                measured = ((det.cx - old_cx) / dt, (det.cy - old_cy) / dt)
                if trk.hits == 1:
                    trk.velocity = measured
                else:
                    trk.velocity = tuple(
                        0.7 * current + 0.3 * previous
                        for current, previous in zip(measured, trk.velocity, strict=True)
                    )
            trk.xyxy = det.xyxy
            trk.hits += 1
            trk.time_since_update = 0
            trk.last_frame = det.frame_id
            trk.history.append(det)
            trk.class_alternatives = alternatives.get(id(det), ())
            del trk.history[: -self.config.class_vote_window]
            self._update_class(trk)

        for ti, trk in enumerate(self._tracks):
            if ti not in matched_trk:
                trk.time_since_update += 1
            trk.age += 1

        for di, det in enumerate(detections):
            if di in matched_det:
                continue
            self._tracks.append(
                _Track(
                    track_id=self._next_id,
                    class_id=det.class_id,
                    class_name=det.class_name,
                    xyxy=det.xyxy,
                    confidence=det.confidence,
                    first_frame=det.frame_id,
                    last_frame=det.frame_id,
                    history=[det],
                    age=1,
                    class_alternatives=alternatives.get(id(det), ()),
                )
            )
            self._next_id += 1

        self._tracks = [t for t in self._tracks if t.time_since_update <= self.config.max_age]
        result: list[TrackedDetection] = []
        for trk in self._tracks:
            last = trk.history[-1]
            candidate = _evidence_source(last) == "open_vocabulary"
            minimum = self.config.open_vocabulary_min_hits if candidate else self.config.min_hits
            if trk.time_since_update > 0 or trk.hits < minimum:
                continue
            metadata = {
                **last.metadata,
                "latest_raw_class": last.class_name,
                "stabilized_class": trk.class_name,
                "temporal_class_disagreement": last.class_id != trk.class_id,
            }
            if candidate:
                metadata.update(
                    classification_candidate_only=True,
                    classification_needs_review=True,
                    classification_status="unverified_text_prompt_prediction",
                )
            if last.class_id != trk.class_id:
                metadata.update(
                    classification_needs_review=True,
                    classification_status="temporal_class_candidate",
                    temporal_classification_reason="latest_class_differs_from_temporal_vote",
                )
                if not metadata.get("classification_reason"):
                    metadata["classification_reason"] = "latest_class_differs_from_temporal_vote"
            stable = Detection(
                frame_id=last.frame_id,
                timestamp=last.timestamp,
                class_id=trk.class_id,
                class_name=trk.class_name,
                confidence=trk.confidence,
                xyxy=trk.xyxy,
                metadata=metadata,
            )
            result.append(
                TrackedDetection(
                    track_id=trk.track_id,
                    detection=stable,
                    hits=trk.hits,
                    raw_detection=last,
                    class_consistency=trk.class_consistency,
                    class_observations=trk.class_observations,
                    class_alternatives=trk.class_alternatives,
                )
            )
        return result

    def _suppress_ambiguous_duplicates(
        self, detections: list[Detection]
    ) -> tuple[list[Detection], dict[int, tuple[Detection, ...]]]:
        """Keep the strongest same-box hypothesis only within explicit groups.

        A person in front of a chair and other overlapping distinct categories
        remain independent. Suppressed hypotheses remain available as evidence.
        """
        groups = [set(group) for group in self.config.ambiguity_groups]
        if not groups:
            return detections, {}
        kept: list[Detection] = []
        alternatives: dict[int, tuple[Detection, ...]] = {}
        for detection in sorted(detections, key=lambda item: item.confidence, reverse=True):
            duplicate = next(
                (
                    previous
                    for previous in kept
                    if previous.class_id != detection.class_id
                    and _evidence_source(previous) == _evidence_source(detection)
                    and any(
                        {previous.class_name, detection.class_name} <= group for group in groups
                    )
                    and _iou(previous.xyxy, detection.xyxy) >= self.config.duplicate_iou
                ),
                None,
            )
            if duplicate is None:
                kept.append(detection)
            else:
                key = id(duplicate)
                alternatives[key] = (*alternatives.get(key, ()), detection)
        return kept, alternatives

    def _predicted_box(self, trk: _Track, frame_id: int) -> tuple[float, float, float, float]:
        if not self.config.motion_prediction or trk.hits < 2:
            return trk.xyxy
        dt = max(0, frame_id - trk.last_frame)
        dx, dy = trk.velocity[0] * dt, trk.velocity[1] * dt
        return (trk.xyxy[0] + dx, trk.xyxy[1] + dy, trk.xyxy[2] + dx, trk.xyxy[3] + dy)

    def _update_class(self, trk: _Track) -> None:
        votes: dict[int, float] = {}
        counts: dict[int, int] = {}
        names: dict[int, str] = {}
        for detection in trk.history:
            cid = detection.class_id
            votes[cid] = votes.get(cid, 0.0) + detection.confidence
            counts[cid] = counts.get(cid, 0) + 1
            names[cid] = detection.class_name
        winner = max(votes, key=lambda cid: (votes[cid], cid == trk.class_id, -cid))
        current_mass = votes.get(trk.class_id, 0.0)
        if (
            winner != trk.class_id
            and counts[winner] >= self.config.class_switch_hits
            and votes[winner] > current_mass * self.config.class_switch_ratio
        ):
            trk.class_id = winner
            trk.class_name = names[winner]
        stable_mass = votes.get(trk.class_id, 0.0)
        trk.class_observations = counts.get(trk.class_id, 0)
        trk.confidence = stable_mass / trk.class_observations if trk.class_observations else 0.0
        total_mass = sum(votes.values())
        trk.class_consistency = stable_mass / total_mass if total_mass else 0.0

    @property
    def tracks(self) -> list[_Track]:
        return list(self._tracks)

    def summary_tracks(self) -> list[dict]:
        return [
            {
                "track_id": trk.track_id,
                "class_id": trk.class_id,
                "class_name": trk.class_name,
                "hits": trk.hits,
                "first_frame": trk.first_frame,
                "last_frame": trk.last_frame,
                "class_consistency": trk.class_consistency,
                "class_observations": trk.class_observations,
            }
            for trk in self._tracks
        ]


def _valid_detection(detection: Detection) -> bool:
    return (
        len(detection.xyxy) == 4
        and all(isfinite(value) for value in detection.xyxy)
        and detection.width > 0
        and detection.height > 0
        and isfinite(detection.confidence)
        and 0 <= detection.confidence <= 1
    )


def _evidence_source(detection: Detection) -> str:
    """Keep rule, review and detector trajectories in separate source groups."""
    if detection.metadata.get("classification_source") == "image_structure_rule":
        return "structure"
    if detection.metadata.get("user_confirmed") is True:
        return "confirmed_review"
    if detection.metadata.get(
        "classification_source"
    ) == "open_vocabulary_model" and detection.metadata.get(
        "open_vocabulary_class_is_novel", detection.class_name not in COCO80_NAMES
    ):
        return "open_vocabulary"
    return "model"


def _maximum_weight_matching(weights: np.ndarray) -> list[tuple[int, int]]:
    """Maximum-cardinality, maximum-score assignment, with unmatched dummies.

    Rectangular Hungarian assignment avoids the greedy failure where the best
    pair consumes the only detection available to a second track. No SciPy
    dependency is needed. Negative weights denote forbidden associations.
    """
    rows, detections = weights.shape
    if not rows or not detections:
        return []
    costs = np.zeros((rows, detections + rows), dtype=float)
    costs[:, :detections] = np.where(weights >= 0, -(rows + 1 + weights), 1e6)
    columns = costs.shape[1]
    u, v = np.zeros(rows + 1), np.zeros(columns + 1)
    assignment = np.zeros(columns + 1, dtype=int)
    for row in range(1, rows + 1):
        assignment[0] = row
        column = 0
        best = np.full(columns + 1, np.inf)
        used = np.zeros(columns + 1, dtype=bool)
        previous = np.zeros(columns + 1, dtype=int)
        while True:
            used[column] = True
            active_row = assignment[column]
            delta, next_column = np.inf, 0
            for candidate in range(1, columns + 1):
                if used[candidate]:
                    continue
                reduced = costs[active_row - 1, candidate - 1] - u[active_row] - v[candidate]
                if reduced < best[candidate]:
                    best[candidate] = reduced
                    previous[candidate] = column
                if best[candidate] < delta:
                    delta, next_column = best[candidate], candidate
            u[assignment[used]] += delta
            v[used] -= delta
            best[~used] -= delta
            column = next_column
            if assignment[column] == 0:
                break
        while column:
            prior = previous[column]
            assignment[column] = assignment[prior]
            column = prior
    return [
        (int(assignment[column] - 1), column - 1)
        for column in range(1, detections + 1)
        if assignment[column] and weights[assignment[column] - 1, column - 1] >= 0
    ]


def _iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return float(inter / union) if union > 0 else 0.0
