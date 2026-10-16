"""SLAM backend factory. Import this from app/CLI — never concrete backends."""

from __future__ import annotations

from visualmap.slam.backends.opencv_vo import create_backend
from visualmap.slam.base import SLAMBackend

__all__ = ["create_backend", "SLAMBackend"]
