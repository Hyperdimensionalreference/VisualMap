"""Small convex-polyhedron operations for bounded visual occupancy hypotheses.

Intersecting image-box viewing cones removes unsupported space from a proxy.
It does not discover hidden surfaces or turn a monocular scale into metres.
"""

from __future__ import annotations

import numpy as np

FRUSTUM_FACES = ((0, 3, 2, 1), (4, 5, 6, 7), (0, 1, 5, 4), (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7))


def _unique(points, epsilon):
    result = []
    for point in points:
        if not any(np.linalg.norm(point - other) <= epsilon for other in result):
            result.append(np.asarray(point, dtype=float))
    return result


def _tolerance(points):
    array = np.asarray(points, dtype=float)
    extent = float(np.linalg.norm(np.ptp(array, axis=0)))
    magnitude = float(np.max(np.abs(array)))
    # Object scale controls geometric tolerance. Absolute world coordinates
    # only contribute their actual floating-point resolution, not a fraction
    # of the arbitrary world origin (which could collapse a small solid).
    return max(extent * 1e-8, abs(float(np.spacing(magnitude))) * 4)


def clip_faces(faces: list[np.ndarray], normal: np.ndarray, offset: float) -> list[np.ndarray]:
    """Keep n dot X + offset >= 0, adding the cut face to keep a closed solid."""
    if not faces:
        return []
    all_points = np.concatenate(faces)
    epsilon = _tolerance(all_points)
    distances = all_points @ normal + offset
    if np.min(distances) >= -epsilon:
        return faces
    if np.max(distances) < -epsilon:
        return []
    clipped, intersections = [], []
    for polygon in faces:
        output = []
        previous = polygon[-1]
        previous_distance = float(normal @ previous + offset)
        for vertex in polygon:
            distance = float(normal @ vertex + offset)
            inside, previous_inside = distance >= -epsilon, previous_distance >= -epsilon
            if inside != previous_inside:
                factor = previous_distance / (previous_distance - distance)
                point = previous + np.clip(factor, 0, 1) * (vertex - previous)
                output.append(point)
                intersections.append(point)
            if inside:
                output.append(vertex)
            previous, previous_distance = vertex, distance
        output = _unique(output, epsilon)
        if len(output) >= 3:
            clipped.append(np.asarray(output))
    cap = _unique(intersections, epsilon)
    if len(cap) >= 3:
        center = np.mean(cap, axis=0)
        axis = np.array([1.0, 0, 0]) if abs(normal[0]) < 0.9 else np.array([0.0, 1, 0])
        first = np.cross(normal, axis)
        first /= np.linalg.norm(first)
        second = np.cross(normal, first)
        cap.sort(key=lambda point: np.arctan2((point - center) @ second, (point - center) @ first))
        clipped.append(np.asarray(cap))
    return clipped


def serialize_shape(faces: list[np.ndarray]) -> tuple[list, list, float]:
    """Return vertices, polygon indices, and rotation-invariant enclosed volume."""
    if not faces:
        return [], [], 0.0
    vertices = _unique(np.concatenate(faces), _tolerance(np.concatenate(faces)))
    if len(vertices) < 4:
        return [], [], 0.0
    center = np.mean(vertices, axis=0)
    indices, volume = [], 0.0
    for face in faces:
        indices.append([int(np.argmin([np.linalg.norm(v - p) for v in vertices])) for p in face])
        for index in range(1, len(face) - 1):
            volume += (
                abs(
                    float(
                        np.dot(
                            face[0] - center,
                            np.cross(face[index] - center, face[index + 1] - center),
                        )
                    )
                )
                / 6
            )
    return np.asarray(vertices).tolist(), indices, volume


def frustum_faces(vertices) -> list[np.ndarray]:
    array = np.asarray(vertices, dtype=float)
    if array.shape != (8, 3) or not np.isfinite(array).all():
        return []
    return [array[list(indices)] for indices in FRUSTUM_FACES]
