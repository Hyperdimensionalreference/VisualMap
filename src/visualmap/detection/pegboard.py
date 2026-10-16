"""Conservative image-structure candidates for perforated surfaces.

The COCO detector has no pegboard output neuron. This module checks actual
pixels for a projective hole lattice and exposes its own rule evidence.
Pixels alone cannot distinguish a pegboard from perforated furniture, a grille
or a displayed dot pattern. Never turn that structure into a verified category
or discard a model prediction because a background pattern overlaps its box.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

import cv2
import numpy as np

from visualmap.detection.types import Detection
from visualmap.slam.types import Frame

PEGBOARD_CLASS_ID = 1000
PERFORATED_PANEL_CLASS_ID = 1001


@dataclass
class PegboardConfig:
    max_dimension: int = 960
    min_holes: int = 45
    min_rows: int = 6
    min_columns: int = 5
    min_pitch_px: float = 4.0
    max_pitch_px: float = 70.0
    min_lattice_inlier_ratio: float = 0.82
    max_lattice_error_pitch: float = 0.18
    min_seed_holes: int = 25
    min_seed_hole_spread: float = 0.3
    min_panel_holes: int = 90
    min_panel_height_ratio: float = 0.28
    min_panel_width_ratio: float = 0.18
    emit_independent_candidates: bool = True

    def __post_init__(self) -> None:
        if self.max_dimension < 160:
            raise ValueError("pegboard max_dimension must be >= 160")
        if self.min_holes < 12 or self.min_rows < 3 or self.min_columns < 3:
            raise ValueError("pegboard needs multiple rows, columns and >= 12 holes")
        if not 0 < self.min_pitch_px < self.max_pitch_px:
            raise ValueError("pegboard pitch bounds must be positive and ordered")
        if not 0 < self.min_lattice_inlier_ratio <= 1:
            raise ValueError("pegboard lattice inlier ratio must be in (0, 1]")
        if not 0 < self.max_lattice_error_pitch < 0.5:
            raise ValueError("pegboard lattice error must be less than half a pitch")
        if self.min_seed_holes < 8 or not 0 < self.min_seed_hole_spread <= 1:
            raise ValueError("pegboard seed evidence needs >= 8 spread-out holes")
        if self.min_panel_holes < self.min_holes:
            raise ValueError("pegboard semantic panel threshold must be >= min_holes")
        if not 0 < self.min_panel_height_ratio <= 1 or not 0 < self.min_panel_width_ratio <= 1:
            raise ValueError("pegboard panel dimensions must be in (0, 1]")


@dataclass
class _Candidate:
    pixels: np.ndarray
    xyxy: tuple[float, float, float, float]
    class_name: str
    score: float
    evidence: dict = field(default_factory=dict)


class PegboardRefiner:
    """Add reviewable surface candidates without replacing model detections."""

    def __init__(self, config: PegboardConfig | None = None) -> None:
        self.config = config or PegboardConfig()

    def refine(self, frame: Frame, detections: list[Detection]) -> list[Detection]:
        candidates = self.candidates(frame.image)
        if not candidates:
            return detections
        result = []
        associated: dict[int, list[dict]] = {}
        for detection in detections:
            chosen = None
            best_count = 0
            for index, candidate in enumerate(candidates):
                pixels = candidate.pixels
                x1, y1, x2, y2 = detection.xyxy
                inside = pixels[
                    (pixels[:, 0] >= x1)
                    & (pixels[:, 0] <= x2)
                    & (pixels[:, 1] >= y1)
                    & (pixels[:, 1] <= y2)
                ]
                if len(inside) < self.config.min_seed_holes:
                    continue
                spread = np.ptp(inside, axis=0) / np.maximum(
                    [detection.width, detection.height],
                    1.0,
                )
                if spread.min() < self.config.min_seed_hole_spread:
                    continue
                if len(inside) > best_count:
                    chosen, best_count = index, len(inside)
            if chosen is None:
                # A surface elsewhere in the image says nothing about this box.
                result.append(detection)
            else:
                result.append(
                    replace(
                        detection,
                        metadata={
                            **detection.metadata,
                            "classification_source": detection.metadata.get(
                                "classification_source", "model_inference"
                            ),
                            "classification_status": "needs_review",
                            "classification_needs_review": True,
                            "classification_structural_conflict": True,
                            "classification_reason": "box_overlaps_perforated_surface",
                            "raw_model_class": detection.class_name,
                            "original_model_class": detection.class_name,
                            "structure_holes_inside_original_box": best_count,
                            "model_box_preserved": True,
                            "user_confirmed": False,
                        },
                    )
                )
                associated.setdefault(chosen, []).append(
                    {
                        "class_id": detection.class_id,
                        "class_name": detection.class_name,
                        "confidence": detection.confidence,
                        "xyxy": list(detection.xyxy),
                        "frame_id": detection.frame_id,
                        "supported_holes_inside_original_box": best_count,
                        "original_box_may_mix_panel_and_foreground": True,
                    }
                )
        for index, candidate in enumerate(candidates):
            seeds = associated.get(index, [])
            if not seeds and not self.config.emit_independent_candidates:
                continue
            evidence = {
                "classification_source": "image_structure_rule",
                "classification_status": "structural_candidate",
                "classification_needs_review": True,
                "classification_reason": "ambiguous_perforated_surface",
                "possible_object_types": [
                    "pegboard", "perforated_furniture", "grille", "displayed_pattern"
                ],
                "model_has_pegboard_class": False,
                "trained_pegboard_model": False,
                "class_semantics_validated": False,
                "user_confirmed": False,
                "rule_score_is_probability": False,
                "structure_score": candidate.score,
                "structure_evidence": candidate.evidence,
                "raw_model_detections": seeds,
                # Overlap is not proof of a shared physical object or category.
                "raw_model_class": None,
                "original_model_class": None,
                "box_source": "visible_hole_array_extent",
                "full_object_boundary_verified": False,
                "physical_object_membership_verified": False,
                "foreground_occlusion_possible": True,
            }
            result.append(
                Detection(
                    frame.index,
                    frame.timestamp,
                    PERFORATED_PANEL_CLASS_ID,
                    candidate.class_name,
                    candidate.score,
                    candidate.xyxy,
                    metadata=evidence,
                )
            )
        return result

    def candidates(self, image: np.ndarray) -> list[_Candidate]:
        gray = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        scale = min(1.0, self.config.max_dimension / max(gray.shape))
        if scale < 1:
            gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        pixels, areas = self._holes(gray)
        if len(pixels) < self.config.min_holes:
            return []
        candidates = []
        for component in self._components(pixels, areas):
            candidate = self._lattice(pixels[component], areas[component], gray, scale)
            if candidate is not None:
                candidates.append(candidate)
        # One large array can split at a foreground lamp. Keep its strongest
        # connected support; merging disconnected patterns could join two objects.
        candidates.sort(key=lambda c: len(c.pixels), reverse=True)
        return candidates[:4]

    @staticmethod
    def _array_inside_box(pixels, box) -> float:
        x1, y1, x2, y2 = box
        return float(
            (
                (pixels[:, 0] >= x1)
                & (pixels[:, 0] <= x2)
                & (pixels[:, 1] >= y1)
                & (pixels[:, 1] <= y2)
            ).mean()
        )

    @staticmethod
    def _enclosed_foreground_panel(gray, xyxy, pitch) -> tuple[bool, dict]:
        """A contrasting, closed foreground outline is ambiguous furniture."""
        x1, y1, x2, y2 = np.rint(xyxy).astype(int)
        height, width = gray.shape
        gap = max(3, int(round(2 * pitch)))
        strips = [
            (
                gray[max(0, y1 - gap) : y1, max(0, x1) : min(width, x2)],
                gray[y1 : min(height, y1 + gap), max(0, x1) : min(width, x2)],
            ),
            (
                gray[y2 : min(height, y2 + gap), max(0, x1) : min(width, x2)],
                gray[max(0, y2 - gap) : y2, max(0, x1) : min(width, x2)],
            ),
            (
                gray[max(0, y1) : min(height, y2), max(0, x1 - gap) : x1],
                gray[max(0, y1) : min(height, y2), x1 : min(width, x1 + gap)],
            ),
            (
                gray[max(0, y1) : min(height, y2), x2 : min(width, x2 + gap)],
                gray[max(0, y1) : min(height, y2), max(0, x2 - gap) : x2],
            ),
        ]
        contrasts = []
        for outer, inner in strips:
            # Black video padding is not a physical object outline.
            outside = outer[outer >= 20]
            inside = inner[inner >= 20]
            if len(outside) < 20 or len(inside) < 20:
                contrasts.append(None)
            else:
                contrasts.append(abs(float(np.median(outside)) - float(np.median(inside))))
        closed_sides = sum(c is not None and c >= 55 for c in contrasts)
        return closed_sides >= 3, {
            "contrasting_outline_sides": closed_sides,
            "boundary_contrast": contrasts,
        }

    def _holes(self, gray: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
        dark = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
        mask = np.uint8(dark > 22) * 255
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        pixels, areas = [], []
        for contour in contours:
            area = float(cv2.contourArea(contour))
            perimeter = float(cv2.arcLength(contour, True))
            x, y, width, height = cv2.boundingRect(contour)
            if not (3 <= area <= 90 and perimeter > 0 and 0.5 <= width / height <= 1.8):
                continue
            if 4 * np.pi * area / perimeter**2 < 0.5:
                continue
            moments = cv2.moments(contour)
            if moments["m00"] <= 0:
                continue
            cx = moments["m10"] / moments["m00"]
            cy = moments["m01"] / moments["m00"]
            # Reject outlines / rings whose middle is brighter than their edge.
            center_value = float(gray[int(round(cy)), int(round(cx))])
            surroundings = gray[max(0, y - 2) : y + height + 2, max(0, x - 2) : x + width + 2]
            if float(np.quantile(surroundings, 0.8)) - center_value < 24:
                continue
            pixels.append((cx, cy))
            areas.append(area)
        return np.float32(pixels).reshape(-1, 2), np.float32(areas)

    def _neighbors(self, pixels: np.ndarray):
        k = min(9, len(pixels))
        index = cv2.flann_Index(
            np.ascontiguousarray(pixels, dtype=np.float32), dict(algorithm=1, trees=1)
        )
        ids, distances = index.knnSearch(np.float32(pixels), k, params={"checks": 64})
        return ids, np.sqrt(np.maximum(distances, 0))

    def _components(self, pixels: np.ndarray, areas: np.ndarray) -> list[np.ndarray]:
        ids, distances = self._neighbors(pixels)
        nearest = distances[:, 1]
        parent = np.arange(len(pixels))

        def root(i):
            while parent[i] != i:
                parent[i] = parent[parent[i]]
                i = parent[i]
            return i

        for i in range(len(pixels)):
            if not self.config.min_pitch_px <= nearest[i] <= self.config.max_pitch_px:
                continue
            for j, distance in zip(ids[i, 1:], distances[i, 1:], strict=True):
                if (
                    0.65 * nearest[i] <= distance <= 1.45 * nearest[i]
                    and 0.4 <= areas[i] / max(areas[j], 1) <= 2.5
                ):
                    parent[root(int(j))] = root(i)
        groups = {}
        for i in range(len(pixels)):
            groups.setdefault(root(i), []).append(i)
        return [np.int64(group) for group in groups.values() if len(group) >= self.config.min_holes]

    def _lattice(self, pixels, areas, gray, scale) -> _Candidate | None:
        shape = gray.shape
        ids, distances = self._neighbors(pixels)
        pitch = float(np.median(distances[:, 1]))
        if not self.config.min_pitch_px <= pitch <= self.config.max_pitch_px:
            return None
        vectors = []
        pairs = []
        for i in range(len(pixels)):
            for j, distance in zip(ids[i, 1:], distances[i, 1:], strict=True):
                if 0.7 * pitch <= distance <= 1.3 * pitch:
                    vectors.append(pixels[j] - pixels[i])
                    pairs.append((i, int(j)))
        if len(vectors) < 2 * self.config.min_holes:
            return None
        vectors = np.float32(vectors)
        angles = np.mod(np.degrees(np.arctan2(vectors[:, 1], vectors[:, 0])), 180)
        histogram, _ = np.histogram(angles, bins=60, range=(0, 180))
        first = int(histogram.argmax()) * 3 + 1.5
        difference = np.abs(((np.arange(60) * 3 + 1.5 - first + 90) % 180) - 90)
        other = histogram.copy()
        other[(difference < 50) | (difference > 130)] = 0
        if other.max() < 0.15 * histogram.max():
            return None
        second = int(other.argmax()) * 3 + 1.5
        axes = np.deg2rad([first, second])
        basis = np.column_stack((np.cos(axes), np.sin(axes)))
        graph = [[] for _ in pixels]
        for (i, j), vector, angle in zip(pairs, vectors, angles, strict=True):
            delta_angle = np.abs(((angle - np.array([first, second]) + 90) % 180) - 90)
            axis = int(delta_angle.argmin())
            if delta_angle[axis] > 18:
                continue
            step = np.zeros(2, dtype=int)
            step[axis] = 1 if np.dot(vector, basis[axis]) > 0 else -1
            graph[i].append((j, step))
        assigned = {}
        best_assignment = {}
        for start in range(len(pixels)):
            if start in assigned:
                continue
            assigned[start] = np.zeros(2, dtype=int)
            queue = [start]
            local = {start: assigned[start]}
            for i in queue:
                for j, step in graph[i]:
                    if j not in assigned:
                        assigned[j] = assigned[i] + step
                        local[j] = assigned[j]
                        queue.append(j)
            if len(local) > len(best_assignment):
                best_assignment = local
        if len(best_assignment) < self.config.min_holes:
            return None
        selected = np.int64(list(best_assignment))
        grid = np.float32([best_assignment[i] for i in selected])
        # Wrong adjacency in ordinary texture cannot satisfy one planar lattice.
        H, inlier_mask = cv2.findHomography(
            grid,
            pixels[selected],
            cv2.RANSAC,
            self.config.max_lattice_error_pitch * pitch,
        )
        if H is None or inlier_mask is None:
            return None
        inliers = inlier_mask.ravel().astype(bool)
        ratio = float(inliers.mean())
        if ratio < self.config.min_lattice_inlier_ratio or inliers.sum() < self.config.min_holes:
            return None
        grid, selected = grid[inliers], selected[inliers]
        seed_holes = len(selected)
        # Foreground lamp bars can disconnect the adjacency graph while the
        # panel's projective lattice continues behind them. Recover omitted
        # *observed* dark-circle pixels only if they agree with the same fitted
        # integer grid. Never invent covered holes or extend a box from no pixels.
        inverse = np.linalg.inv(H)
        grid_all = cv2.perspectiveTransform(pixels.reshape(-1, 1, 2), inverse).reshape(-1, 2)
        rounded_grid = np.rint(grid_all).astype(np.float32)
        projected_all = cv2.perspectiveTransform(
            rounded_grid.reshape(-1, 1, 2),
            H,
        ).reshape(-1, 2)
        error_all = np.linalg.norm(projected_all - pixels, axis=1)
        valid_all = error_all <= self.config.max_lattice_error_pitch * pitch
        median_area = float(np.median(areas[selected]))
        valid_all &= (areas >= 0.4 * median_area) & (areas <= 2.5 * median_area)
        cells = {}
        for index in np.argsort(error_all):
            if valid_all[index]:
                cell = tuple(rounded_grid[index].astype(int))
                cells.setdefault(cell, int(index))
        recovered = np.int64(list(cells.values()))
        if len(recovered) > seed_holes:
            selected = recovered
            grid = rounded_grid[selected]
            refined_h, _ = cv2.findHomography(grid, pixels[selected], 0)
            if refined_h is not None:
                H = refined_h
        rows, columns = len(np.unique(grid[:, 0])), len(np.unique(grid[:, 1]))
        if min(rows, columns) < min(self.config.min_rows, self.config.min_columns):
            return None
        if max(rows, columns) < max(self.config.min_rows, self.config.min_columns):
            return None
        centers = pixels[selected]
        predicted = cv2.perspectiveTransform(grid.reshape(-1, 1, 2), H).reshape(-1, 2)
        errors = np.linalg.norm(predicted - centers, axis=1)
        diameter = 2 * np.sqrt(float(np.median(areas[selected])) / np.pi)
        diameter_cv = float(np.std(np.sqrt(areas[selected])) / np.mean(np.sqrt(areas[selected])))
        if not 2.0 <= pitch / diameter <= 9.0 or diameter_cv > 0.4:
            return None
        xyxy = np.r_[centers.min(axis=0) - 0.4 * pitch, centers.max(axis=0) + 0.4 * pitch]
        xyxy[[0, 2]] = np.clip(xyxy[[0, 2]], 0, shape[1])
        xyxy[[1, 3]] = np.clip(xyxy[[1, 3]], 0, shape[0])
        height_ratio = float((xyxy[3] - xyxy[1]) / shape[0])
        width_ratio = float((xyxy[2] - xyxy[0]) / shape[1])
        enclosed_foreground, panel_context = self._enclosed_foreground_panel(gray, xyxy, pitch)
        large_background_panel = (
            len(centers) >= self.config.min_panel_holes
            and height_ratio >= self.config.min_panel_height_ratio
            and width_ratio >= self.config.min_panel_width_ratio
            and rows >= 7
            and columns >= 7
            and not enclosed_foreground
        )
        name = "perforated_panel"
        score = float(np.clip(0.65 + 0.2 * ratio - 0.5 * np.median(errors) / pitch, 0.5, 0.88))
        hull = cv2.convexHull(centers / scale).reshape(-1, 2)
        return _Candidate(
            centers / scale,
            tuple(float(value) for value in xyxy / scale),
            name,
            score,
            {
                "method": "dark_round_holes_projective_lattice",
                "hole_count": len(centers),
                "rows": rows,
                "columns": columns,
                "lattice_seed_holes": seed_holes,
                "recovered_observed_lattice_holes": len(centers) - seed_holes,
                "lattice_inlier_ratio": ratio,
                "median_lattice_error_px": float(np.median(errors) / scale),
                "median_lattice_error_pitch": float(np.median(errors) / pitch),
                "hole_pitch_px": pitch / scale,
                "hole_diameter_px": diameter / scale,
                "hole_size_coefficient_of_variation": diameter_cv,
                "large_background_panel": bool(large_background_panel),
                "foreground_panel_ambiguous": enclosed_foreground,
                "panel_context": panel_context,
                "support_polygon": hull.astype(float).tolist(),
                "semantic_limit": (
                    "Image structure does not identify the physical object type; "
                    "furniture, grilles and displayed patterns can share this lattice."
                ),
            },
        )
