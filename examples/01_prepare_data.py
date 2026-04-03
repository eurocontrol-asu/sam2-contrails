"""Prepare GVCCS data: extract per-video frames from the flat image folder.

The GVCCS dataset stores all frames as ``image_YYYYMMDDHHMMSS.jpg`` in a
single flat directory. This script reads ``annotations.json`` to discover
which images belong to each video and symlinks them into per-video folders
named by COCO integer video ID, with sequential frame names::

    data/frames/00001/00000.jpg
    data/frames/00001/00001.jpg
    ...

This naming convention matches the output of ``contrailtrack generate-prompts``
(which also uses COCO integer video IDs and sequential frame indices), so frames
and prompts align exactly at inference time.

Frames are sorted by their COCO ``time`` field (chronological order), so
``00000.jpg`` is always the earliest frame.

Usage::

    # Prepare all videos
    uv run python examples/01_prepare_data.py /path/to/GVCCS/test

    # Prepare a single video by its timestamp name
    uv run python examples/01_prepare_data.py /path/to/GVCCS/test --video 20230930055430_20230930075430
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import structlog
import typer

log = structlog.get_logger()

app = typer.Typer(add_completion=False)


@app.command()
def main(
    gvccs_dir: Annotated[
        Path,
        typer.Argument(help="GVCCS test directory (contains annotations.json, images/, parquet/)."),
    ],
    output: Annotated[
        Path,
        typer.Option("--output", "-o", help="Output directory for per-video frame folders."),
    ] = Path("data/frames"),
    video: Annotated[
        str | None,
        typer.Option(help="Prepare only this video (e.g. 20230930055430_20230930075430). Default: all."),
    ] = None,
    copy: Annotated[
        bool,
        typer.Option(help="Copy files instead of creating symlinks."),
    ] = False,
) -> None:
    """Extract per-video frame folders from the flat GVCCS image directory."""
    import shutil

    annotations_path = gvccs_dir / "annotations.json"
    images_dir = gvccs_dir / "images"

    if not annotations_path.exists():
        log.error("annotations_not_found", path=str(annotations_path))
        raise typer.Exit(code=1)
    if not images_dir.exists():
        log.error("images_dir_not_found", path=str(images_dir))
        raise typer.Exit(code=1)

    with open(annotations_path) as f:
        ann = json.load(f)

    # Build video_id → video_name from start/stop timestamps
    video_meta = {}
    for v in ann["videos"]:
        start = v["start"].replace("-", "").replace(":", "").replace("T", "")
        stop = v["stop"].replace("-", "").replace(":", "").replace("T", "")
        video_meta[v["id"]] = {"name": f"{start}_{stop}", "length": v["length"]}

    # Group images by video
    images_by_video: dict[int, list[dict]] = {}
    for img in ann["images"]:
        images_by_video.setdefault(img["video_id"], []).append(img)

    for vid_id, imgs in sorted(images_by_video.items()):
        meta = video_meta[vid_id]
        video_name = meta["name"]

        if video and video_name != video:
            continue

        imgs_sorted = sorted(imgs, key=lambda x: x["time"])
        # Use COCO integer video ID as folder name (matches generate-prompts output)
        out_dir = output / str(vid_id).zfill(5)
        out_dir.mkdir(parents=True, exist_ok=True)

        n_existing = len(list(out_dir.glob("*.jpg")))
        if n_existing == len(imgs_sorted):
            log.info("skipping_existing", video=video_name, video_id=str(vid_id).zfill(5), frames=n_existing)
            continue

        for frame_idx, img in enumerate(imgs_sorted):
            src = images_dir / img["file_name"]
            # Sequential zero-padded name: 00000.jpg, 00001.jpg, ...
            dst = out_dir / f"{str(frame_idx).zfill(5)}.jpg"

            if dst.exists():
                continue

            if copy:
                shutil.copy2(src, dst)
            else:
                dst.symlink_to(src.resolve())

        log.info("prepared", video=video_name, video_id=str(vid_id).zfill(5), frames=len(imgs_sorted))

    # Symlink parquet files for easy access
    parquet_out = output.parent / "parquet"
    src_parquet = gvccs_dir / "parquet"
    if src_parquet.exists():
        parquet_out.mkdir(parents=True, exist_ok=True)
        if video:
            src = src_parquet / f"{video}.parquet"
            dst = parquet_out / f"{video}.parquet"
            if src.exists() and not dst.exists():
                dst.symlink_to(src.resolve())
                log.info("parquet_linked", file=f"{video}.parquet")
        else:
            for src in sorted(src_parquet.glob("*.parquet")):
                dst = parquet_out / src.name
                if not dst.exists():
                    dst.symlink_to(src.resolve())
            log.info("parquets_linked", path=str(parquet_out))

    log.info("done", output=str(output))


if __name__ == "__main__":
    app()
