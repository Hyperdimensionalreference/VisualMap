"""Explicit download of the Apache-2.0 OpenCV Zoo YOLOX asset, hash checked."""

from __future__ import annotations

import argparse
import hashlib
import tempfile
from pathlib import Path
from urllib.parse import urlsplit
from urllib.request import Request, urlopen

URL = (
    "https://media.githubusercontent.com/media/opencv/opencv_zoo/main/"
    "models/object_detection_yolox/object_detection_yolox_2022nov.onnx"
)
SHA256 = "c5c2d13e59ae883e6af3b45daea64af4833a4951c92d116ec270d9ddbe998063"
SIZE = 35_858_002
DEFAULT = Path(__file__).resolve().parents[1] / "models/object_detection_yolox_2022nov.onnx"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--path", type=Path, default=DEFAULT)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    target = args.path
    if target.exists():
        if target.is_symlink() or not target.is_file():
            raise SystemExit("Model target must be a regular file")
        if hashlib.sha256(target.read_bytes()).hexdigest() != SHA256:
            raise SystemExit("Existing model hash differs; no file was replaced")
        print("YOLOX model SHA-256 verified")
        return
    if args.check_only:
        raise SystemExit("Model is absent; --check-only performs no network request")
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=target.parent, prefix=".yolox-", suffix=".tmp", delete=False
        ) as stream:
            temporary = Path(stream.name)
            digest, size = hashlib.sha256(), 0
            with urlopen(
                Request(URL, headers={"User-Agent": "VisualMap-model-download"}), timeout=60
            ) as response:
                final = urlsplit(response.geturl())
                if final.scheme != "https" or final.hostname != "media.githubusercontent.com":
                    raise ValueError("Unexpected download destination")
                while chunk := response.read(1024 * 1024):
                    size += len(chunk)
                    if size > SIZE:
                        raise ValueError("Unexpected model size")
                    digest.update(chunk)
                    stream.write(chunk)
            if size != SIZE or digest.hexdigest() != SHA256:
                raise ValueError("Model size or SHA-256 mismatch; download not installed")
        temporary.replace(target)
        print("YOLOX model downloaded and SHA-256 verified; Apache-2.0 notices in LICENSES/")
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
