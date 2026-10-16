from pathlib import Path

import numpy as np
import pytest

from visualmap.export.reconstruction import export_reconstruction, load_reconstruction
from visualmap.slam.types import CameraIntrinsics, CameraPose, MapPoint, MapResult, Trajectory


def test_roundtrip_keeps_rotation_observed_landmarks_and_pose_validity(tmp_path: Path):
    pose = CameraPose.from_rt(
        [[0, 0, 1], [0, 1, 0], [-1, 0, 0]], [2, 3, 4], frame_id=9, timestamp=0.3
    )
    result = MapResult(
        trajectory=Trajectory([pose]),
        map_points=[MapPoint(id=12, xyz=[1, 2, 5], observations={0: 77, 9: 77})],
        intrinsics=CameraIntrinsics(fx=100, fy=110, cx=50, cy=40, width=100, height=80),
        metadata={"frame_quality": {"9": {"pose_valid": True}}, "scale_consistent": True},
    )
    loaded = load_reconstruction(export_reconstruction(result, tmp_path / "空间" / "map.json"))
    np.testing.assert_allclose(loaded.trajectory.poses[0].matrix, pose.matrix)
    np.testing.assert_allclose(loaded.map_points[0].xyz, [1, 2, 5])
    assert loaded.map_points[0].observations == {0: 77, 9: 77}
    assert loaded.metadata == result.metadata
    assert loaded.intrinsics == result.intrinsics


def test_reconstruction_rejects_wrong_pose_convention(tmp_path: Path):
    path = tmp_path / "wrong.json"
    path.write_text('{"schema_version":1,"pose_convention":"T_camera<-world"}')
    with pytest.raises(ValueError, match="pose convention"):
        load_reconstruction(path)
