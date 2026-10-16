import csv
from pathlib import Path

import numpy as np
import pytest

from visualmap.export import export_all, export_trajectory_csv, export_trajectory_txt
from visualmap.slam.types import CameraIntrinsics, CameraPose, MapPoint, MapResult, Trajectory


def _demo_result() -> MapResult:
    traj = Trajectory()
    traj.append(CameraPose.from_rt(np.eye(3), [0, 0, 0], frame_id=0, timestamp=0.0))
    traj.append(CameraPose.from_rt(np.eye(3), [1, 0, 0], frame_id=1, timestamp=0.1))
    points = [MapPoint(id=0, xyz=[0.5, 0.1, 2.0], color=(10, 20, 30))]
    return MapResult(
        trajectory=traj,
        map_points=points,
        intrinsics=CameraIntrinsics(fx=100, fy=100, cx=50, cy=40, width=100, height=80),
        metadata={"stage": "test"},
    )


def test_export_all(tmp_path: Path):
    result = _demo_result()
    written = export_all(result, tmp_path, run_name="t1")
    assert set(written) == {"trajectory_csv", "trajectory_txt", "map_ply", "report"}
    assert (tmp_path / "t1" / "trajectory.csv").is_file()
    assert (tmp_path / "t1" / "trajectory.txt").is_file()
    assert (tmp_path / "t1" / "map_points.ply").is_file()
    assert (tmp_path / "t1" / "run_report.json").is_file()

    csv_text = (tmp_path / "t1" / "trajectory.csv").read_text(encoding="utf-8")
    assert csv_text.splitlines()[0].startswith("frame_id,timestamp")
    assert len(csv_text.strip().splitlines()) == 3  # header + 2 poses

    ply = (tmp_path / "t1" / "map_points.ply").read_text(encoding="utf-8")
    assert "element vertex 1" in ply
    assert "end_header" in ply


def test_export_rejects_escaping_run_name(tmp_path: Path):
    with pytest.raises(ValueError, match="run_name"):
        export_all(_demo_result(), tmp_path / "out", run_name="../escape")
    assert not (tmp_path / "escape").exists()


def test_csv_and_tum_quaternion_orders(tmp_path: Path):
    rotation = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=float)
    result = MapResult(
        trajectory=Trajectory([CameraPose.from_rt(rotation, [1, 2, 3], frame_id=7)]),
    )
    csv_path = export_trajectory_csv(result, tmp_path / "trajectory.csv")
    txt_path = export_trajectory_txt(result, tmp_path / "trajectory.txt")
    with csv_path.open(newline="", encoding="utf-8") as handle:
        row = next(csv.DictReader(handle))
    assert float(row["qw"]) == pytest.approx(2**-0.5)
    assert float(row["qz"]) == pytest.approx(2**-0.5)
    tum = [float(value) for value in txt_path.read_text(encoding="utf-8").split()]
    assert tum[4:8] == pytest.approx([0, 0, 2**-0.5, 2**-0.5], abs=1e-8)


def test_independent_submaps_are_separated_in_ply_tum_and_labelled_in_csv(tmp_path: Path):
    import json

    result = _demo_result()
    result.map_points.append(MapPoint(1, [0, 0, 9], observations={1: 1}))
    result.map_points[0].observations = {0: 0}
    result.metadata.update(
        submaps=[{"world_frame_id": "a"}, {"world_frame_id": "b"}],
        frame_quality={"0": {"world_frame_id": "a", "pose_valid": True},
                       "1": {"world_frame_id": "b", "pose_valid": True}},
    )
    export_all(result, tmp_path, "segmented")
    output = tmp_path / "segmented"
    manifest = json.loads((output / "map_points.coordinate_frames.json").read_text())
    assert [row["world_frame_id"] for row in manifest["maps"]] == ["a", "b"]
    assert all(row["points"] == 1 for row in manifest["maps"])
    assert "element vertex 1" in (output / "map_points.ply").read_text()
    assert "element vertex 1" in (output / "map_points_submap_2.ply").read_text()
    with (output / "trajectory.csv").open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert [row["world_frame_id"] for row in rows] == ["a", "b"]
    assert [row["pose_valid"] for row in rows] == ["True", "True"]
    assert (output / "trajectory.txt").read_text().startswith("# Independent submaps")
    assert len((output / "trajectory_submap_1.txt").read_text().strip().splitlines()) == 1
    assert len((output / "trajectory_submap_2.txt").read_text().strip().splitlines()) == 1
