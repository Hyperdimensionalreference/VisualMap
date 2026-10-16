"""Export a self-contained spatial viewer with synchronized video evidence."""

from __future__ import annotations

import base64
import gzip
import hashlib
import json
from pathlib import Path

from visualmap.mapping.semantic_map import SemanticMap
from visualmap.utils.logging import get_logger
from visualmap.utils.paths import run_directory, validate_run_name

logger = get_logger(__name__)
__all__ = ["export_semantic_map_html", "render_viewer_payload", "viewer_payload_from_html"]

PACK_THRESHOLD_BYTES = 128 * 1024


def viewer_payload_from_html(html: str) -> dict:
    """Read both historical plain payloads and new lossless packed exports."""
    html = html.replace("\r\n", "\n")
    marker = 'const VISUALMAP_GZIP_DATA = "'
    if marker in html:
        packed = html.split(marker, 1)[1].split('";', 1)[0]
        return json.loads(gzip.decompress(base64.b64decode(packed, validate=True)))
    return json.loads(html.split("const DATA = ", 1)[1].split(";\n", 1)[0])


def _serialized_payload(payload: dict) -> tuple[str, bool]:
    # Human-sized fixtures stay readable; large result packages use compact,
    # lossless data. All observations and original floating-point values remain.
    readable = json.dumps(payload, ensure_ascii=False, allow_nan=False).replace("<", "\\u003c")
    large = len(readable.encode("utf-8")) >= PACK_THRESHOLD_BYTES
    data_json = (
        (
            json.dumps(payload, ensure_ascii=False, allow_nan=False, separators=(",", ":")).replace(
                "<", "\\u003c"
            )
        )
        if large
        else readable
    )
    return data_json, large


def render_viewer_payload(payload: dict) -> str:
    """Package current assets around saved data without recalculating recognition."""
    data_json, large = _serialized_payload(payload)
    assets = Path(__file__).parent
    html = (assets / "viewer.html").read_text(encoding="utf-8")
    style = (assets / "viewer.css").read_text(encoding="utf-8")
    font = assets / "fonts/Geist-Variable.woff2"
    if font.is_file():
        encoded_font = base64.b64encode(font.read_bytes()).decode("ascii")
        style = (
            "@font-face{font-family:Geist;font-style:normal;font-weight:100 900;"
            "font-display:swap;src:url(data:font/woff2;base64,"
            + encoded_font
            + ") format('woff2')}\n"
        ) + style
    html = html.replace("__STYLE__", style)
    motion_path = assets / "viewer_motion.js"
    motion = motion_path.read_text(encoding="utf-8") if motion_path.is_file() else ""
    occupancy_script = (assets / "viewer_occupancy.js").read_text(encoding="utf-8")
    html = html.replace(
        "__SCRIPT__",
        motion
        + "\n"
        + occupancy_script
        + "\n"
        + (assets / "viewer.js").read_text(encoding="utf-8"),
    )
    if large:
        packed = base64.b64encode(gzip.compress(data_json.encode("utf-8"), mtime=0)).decode("ascii")
        loader = (
            '<script>\n(async function(){\nconst VISUALMAP_GZIP_DATA = "' + packed + '";\n'
            "if(typeof DecompressionStream!=='function')throw new Error('browser');\n"
            "const packedBytes=Uint8Array.from(atob(VISUALMAP_GZIP_DATA),c=>c.charCodeAt(0));\n"
            "const unpacked=new Blob([packedBytes]).stream()"
            ".pipeThrough(new DecompressionStream('gzip'));\n"
            "const DATA=JSON.parse(await new Response(unpacked).text());\n"
        )
        html = html.replace("<script>const DATA = __DATA__;\n", loader, 1)
        html = html.replace(
            "</script></body></html>",
            "\n})().catch(()=>{const notice=document.getElementById('alert');"
            "if(notice){notice.hidden=false;notice.textContent='结果尚未加载，请使用较新版浏览器，或重新从本机工作台打开。';}});"
            "\n</script></body></html>",
            1,
        )
        if "__DATA__" in html:
            raise ValueError("viewer template is missing the data bootstrap marker")
    else:
        html = html.replace("__DATA__", data_json)
    return html


def export_semantic_map_html(
    semantic_map: SemanticMap,
    output_dir: str | Path,
    run_name: str = "run",
    title: str = "VisualMap 三维空间",
    video_filename: str | None = None,
) -> Path:
    """Embed exported data and packaged assets; no network required by viewer."""
    out = run_directory(output_dir, run_name)
    out.mkdir(parents=True, exist_ok=True)
    if video_filename is not None:
        validate_run_name(video_filename)
    payload = {
        "run_id": run_name,
        "title": title,
        "summary": semantic_map.summary(),
        "objects": [obj.to_dict() for obj in semantic_map.objects],
        "map_points": [mp.xyz.tolist() for mp in semantic_map.map_points[:4000]],
        "map_point_world_frame_ids": semantic_map.map_point_world_frame_ids[:4000],
        "trajectory": semantic_map.trajectory_xyz,
        "trajectory_timestamps": semantic_map.trajectory_timestamps,
        "trajectory_frame_ids": semantic_map.trajectory_frame_ids,
        "trajectory_rotations": getattr(semantic_map, "trajectory_rotations", []),
        "trajectory_world_frame_ids": getattr(semantic_map, "trajectory_world_frame_ids", []),
        "object_observations": getattr(semantic_map, "object_observations", []),
        "video_filename": video_filename,
    }
    data_json, _ = _serialized_payload(payload)
    html = render_viewer_payload(payload)
    path = out / "semantic_map.html"
    path.write_text(html, encoding="utf-8")
    occupancy_records = [
        {"track_id": obj.track_id, **snapshot["occupancy"]}
        for obj in semantic_map.objects
        for snapshot in obj.metadata.get("position_snapshots", [])
        if "occupancy" in snapshot
    ]
    (out / "occupancy.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "export_id": hashlib.sha256(data_json.encode("utf-8")).hexdigest(),
                "run_name": run_name,
                "coordinate_scope": "within_same_world_frame_id_in_this_export_only",
                "pose_convention": "T_world<-camera",
                "scale": "monocular_arbitrary",
                "units": "map_units",
                "volume_units": "map_units3",
                "world_axes": "opencv_y_down",
                "volume_verified": False,
                "free_space_certified": False,
                "policy": (
                    "past_trusted_views_refine_bounded_proxy_not_certified_collision_geometry"
                ),
                "proxy_volume_field": "proxy_volume_map_units3",
                "broad_phase_aabb_volume_field": "aabb_volume_map_units3",
                "sensitivity_is_confidence_interval": False,
                "objects": [
                    {
                        "track_id": obj.track_id,
                        "class_name": obj.class_name,
                        "occupancy": obj.metadata.get("occupancy"),
                    }
                    for obj in semantic_map.objects
                ],
                "snapshots": occupancy_records,
            },
            ensure_ascii=False,
            separators=(",", ":"),
            allow_nan=False,
        ),
        encoding="utf-8",
    )
    logger.info("Wrote spatial viewer %s", path)
    return path
