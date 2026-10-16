"""Open3D visualization for trajectory + sparse map + semantic objects.

Open3D is an optional extra (`pip install 'visualmap[viz]'`).
HTML viewer (`export_semantic_map_html`) works without Open3D.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from visualmap.mapping.semantic_map import SemanticMap
from visualmap.slam.types import CameraPose, MapResult, Trajectory
from visualmap.utils.logging import get_logger

logger = get_logger(__name__)

__all__ = [
    "open3d_available",
    "build_geometry",
    "build_semantic_geometry",
    "show_map",
    "show_semantic_map",
    "save_map_view",
    "export_semantic_map_html",
]


def open3d_available() -> bool:
    try:
        import open3d  # noqa: F401

        return True
    except ImportError:
        return False


def build_geometry(result: MapResult, semantic_objects=None):
    """Build (point_cloud, trajectory_lineset, camera_markers, coord_frame, object_mesh)."""
    import open3d as o3d

    pts = (
        np.stack([mp.xyz for mp in result.map_points], axis=0)
        if result.map_points
        else np.zeros((0, 3))
    )
    pcd = o3d.geometry.PointCloud()
    if pts.shape[0] > 0:
        pcd.points = o3d.utility.Vector3dVector(pts)
        pcd.paint_uniform_color([0.2, 0.75, 0.95])

    lineset = o3d.geometry.LineSet()
    positions = result.trajectory.positions()
    if positions.shape[0] >= 2:
        lineset.points = o3d.utility.Vector3dVector(positions)
        lineset.lines = o3d.utility.Vector2iVector(
            [[i, i + 1] for i in range(positions.shape[0] - 1)]
        )
        lineset.paint_uniform_color([1.0, 0.55, 0.1])

    markers = o3d.geometry.TriangleMesh()
    if positions.shape[0] > 0:
        step = max(1, positions.shape[0] // 20)
        for i in range(0, positions.shape[0], step):
            s = o3d.geometry.TriangleMesh.create_sphere(radius=0.08)
            s.translate(positions[i])
            s.paint_uniform_color([0.95, 0.25, 0.25])
            markers += s

    frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5)
    obj_mesh = build_semantic_geometry(semantic_objects or [])
    return pcd, lineset, markers, frame, obj_mesh


def build_semantic_geometry(semantic_objects):
    """Spheres at semantic object world positions (if any)."""
    import open3d as o3d

    mesh = o3d.geometry.TriangleMesh()
    palette = [
        [0.91, 0.36, 0.36],
        [0.94, 0.70, 0.28],
        [0.35, 0.78, 0.55],
        [0.45, 0.55, 0.95],
    ]
    for i, obj in enumerate(semantic_objects):
        xyz = getattr(obj, "xyz", None)
        if xyz is None:
            continue
        s = o3d.geometry.TriangleMesh.create_sphere(radius=0.18)
        s.translate(np.asarray(xyz, dtype=np.float64).reshape(3))
        s.paint_uniform_color(palette[i % len(palette)])
        mesh += s
    return mesh


def show_map(
    result: MapResult,
    window_name: str = "VisualMap",
    semantic_objects=None,
) -> None:
    """Open an interactive Open3D window. Blocks until closed."""
    if not open3d_available():
        raise RuntimeError(
            "Open3D is not installed. Install with: pip install 'visualmap[viz]'"
        )
    import open3d as o3d

    pcd, lineset, markers, frame, obj_mesh = build_geometry(result, semantic_objects)
    geoms = [g for g in [pcd, lineset, markers, frame, obj_mesh] if g is not None]
    o3d.visualization.draw_geometries(
        geoms,
        window_name=window_name,
        width=1280,
        height=800,
        point_show_normal=False,
    )


def show_semantic_map(semantic_map: SemanticMap, window_name: str = "VisualMap Semantic") -> None:
    traj = Trajectory()
    for xyz in semantic_map.trajectory_xyz:
        traj.append(CameraPose.from_rt(np.eye(3), xyz))
    result = MapResult(trajectory=traj, map_points=semantic_map.map_points)
    show_map(result, window_name=window_name, semantic_objects=semantic_map.objects)


def save_map_view(result: MapResult, path: str | Path) -> Path:
    """Save non-interactive PLY bundle (point cloud + trajectory)."""
    if not open3d_available():
        raise RuntimeError(
            "Open3D is not installed. Install with: pip install 'visualmap[viz]'"
        )
    import open3d as o3d

    pcd, lineset, _markers, _frame, _objs = build_geometry(result)
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    o3d.io.write_point_cloud(str(out.with_suffix(".ply")), pcd)
    if lineset is not None and len(lineset.points) > 0:
        o3d.io.write_line_set(str(out.with_name(out.stem + "_trajectory.ply")), lineset)
    logger.info("Saved Open3D view assets near %s", out)
    return out


def export_semantic_map_html(
    semantic_map: SemanticMap,
    output_dir: str | Path,
    run_name: str = "run",
    title: str = "VisualMap 语义地图",
    video_filename: str | None = None,
) -> Path:
    from visualmap.visualization.semantic_html import export_semantic_map_html as _export

    return _export(
        semantic_map,
        output_dir,
        run_name=run_name,
        title=title,
        video_filename=video_filename,
    )
