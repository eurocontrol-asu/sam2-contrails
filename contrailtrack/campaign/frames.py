"""Create per-video symlink directories from day-long image folders."""

from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path

import structlog

log = structlog.get_logger()

_IMAGE_RE = re.compile(r"^image_(\d{14})\.jpe?g$")


def create_frame_symlinks(
    images_dir: Path,
    output_dir: Path,
    video_id: str,
    start: datetime,
    stop: datetime,
) -> int:
    """Create symlinks for images within [start, stop), stripping the ``image_`` prefix.

    Source: ``images_dir/image_YYYYMMDDHHMMSS.jpg``
    Target: ``output_dir/video_id/YYYYMMDDHHMMSS.jpg``
    """
    images_dir = Path(images_dir)
    video_dir = Path(output_dir) / video_id
    video_dir.mkdir(parents=True, exist_ok=True)

    count = 0
    for img_path in sorted(images_dir.iterdir()):
        m = _IMAGE_RE.match(img_path.name)
        if not m:
            continue

        ts_str = m.group(1)
        ts = datetime.strptime(ts_str, "%Y%m%d%H%M%S")

        if ts < start or ts >= stop:
            continue

        link = video_dir / f"{ts_str}.jpg"
        if not link.exists():
            link.symlink_to(img_path.resolve())
        count += 1

    log.info("frame_symlinks_created", video_id=video_id, count=count)
    return count
