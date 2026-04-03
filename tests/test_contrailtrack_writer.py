"""Tests for contrailtrack.prompts.writer — prompt PNG generation.

Unit tests build synthetic raw CoCiP-format DataFrames (lat/lon space).
The writer projects them in-memory before writing prompt PNGs.

Integration tests use the real GVCCS dataset and are skipped when unavailable.

Writer tests require geopandas, affine, and rasterio. Tests are skipped
automatically when these packages are not installed in the current environment.
"""

import json
import os
import tempfile
import unittest
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

# Integration tests use real CoCiP + GVCCS data. Set these env vars to enable:
#   export COCIP_DATA_DIR=/path/to/cocip/data_model_levels/test
#   export GVCCS_ANNOTATIONS=/path/to/GVCCS/test/annotations.json
_cocip_data_dir = os.environ.get("COCIP_DATA_DIR", "")
_gvccs_annotations = os.environ.get("GVCCS_ANNOTATIONS", "")
REAL_CONTRAIL_DIR = Path(_cocip_data_dir) if _cocip_data_dir else Path("__nonexistent__")
REAL_ANNOTATIONS = Path(_gvccs_annotations) if _gvccs_annotations else Path("__nonexistent__")

real_data_available = _cocip_data_dir != "" and _gvccs_annotations != "" and REAL_CONTRAIL_DIR.exists() and REAL_ANNOTATIONS.exists()

try:
    import geopandas  # noqa
    import affine     # noqa
    geo_deps_available = True
except ImportError:
    geo_deps_available = False


# ─── helpers ──────────────────────────────────────────────────────────────────

# GVCCS camera position — contrail waypoints must be near here to project in-frame
_CAM_LAT = 48.600518
_CAM_LON = 2.346795


def _make_synthetic_contrail_df(video_name: str, n_waypoints: int = 4) -> pd.DataFrame:
    """Create a minimal raw CoCiP/DryAdvection-format DataFrame (lat/lon space).

    Waypoints are placed near the GVCCS camera so they project within the 1024×1024
    image after MiniProjector transformation.
    """
    start_str = video_name.split("_")[0]
    start = datetime.strptime(start_str, "%Y%m%d%H%M%S")
    t0 = start + timedelta(minutes=1)
    formation = t0 - timedelta(minutes=2)

    rows = []
    for fid in ["flight_0", "flight_1"]:
        for wp in range(n_waypoints):
            rows.append({
                "flight_id": fid,
                "time": pd.Timestamp(t0),
                "formation_time": pd.Timestamp(formation),
                "longitude": _CAM_LON - 0.05 + wp * 0.04,
                "latitude": _CAM_LAT - 0.05 + wp * 0.02,
                "level": 250.0,   # hPa → ~34 000 ft
                "waypoint": wp,
                "width": 2000.0,
            })
    return pd.DataFrame(rows)


def _write_contrail_parquet(tmp: Path, video_name: str) -> Path:
    """Write a synthetic contrail DataFrame to a parquet file."""
    df = _make_synthetic_contrail_df(video_name)
    path = tmp / f"{video_name}.parquet"
    df.to_parquet(path)
    return path


def _make_synthetic_coco(tmp: Path, video_name: str, h: int = 1024, w: int = 1024) -> Path:
    """Create a minimal COCO JSON with one video and two frames."""
    tmp.mkdir(parents=True, exist_ok=True)
    start_str = video_name.split("_")[0]
    start = datetime.strptime(start_str, "%Y%m%d%H%M%S")
    t0 = start + timedelta(minutes=1)
    t1 = start + timedelta(minutes=2)

    data = {
        "info": {},
        "videos": [{"id": 1, "start": start.isoformat(), "stop": (start + timedelta(hours=1)).isoformat()}],
        "images": [
            {"id": 1, "video_id": 1, "file_name": "frame_0.jpg", "height": h, "width": w, "time": t0.isoformat()},
            {"id": 2, "video_id": 1, "file_name": "frame_1.jpg", "height": h, "width": w, "time": t1.isoformat()},
        ],
        "annotations": [
            {
                "id": 1, "image_id": 1, "category_id": 1, "flight_id": "flight_0",
                "segmentation": [[10, 10, 50, 10, 50, 30, 10, 30]],
                "area": 800, "bbox": [10, 10, 40, 20], "iscrowd": 0,
            }
        ],
        "categories": [{"id": 1, "name": "contrail"}],
    }

    ann_path = tmp / "annotations.json"
    with open(ann_path, "w") as f:
        json.dump(data, f)
    return ann_path


def _make_jpeg_frames_timestamped(directory: Path, video_name: str, n: int = 2,
                                  h: int = 1024, w: int = 1024):
    """Write JPEG frames named as timestamps matching the contrail data."""
    directory.mkdir(parents=True, exist_ok=True)
    start_str = video_name.split("_")[0]
    start = datetime.strptime(start_str, "%Y%m%d%H%M%S")
    img = Image.fromarray(np.zeros((h, w, 3), dtype=np.uint8))
    for i in range(n):
        ts = start + timedelta(minutes=1 + i)
        img.save(directory / f"{ts.strftime('%Y%m%d%H%M%S')}.jpg")


# ─── generate_prompts (COCO batch mode) ───────────────────────────────────────

@unittest.skipUnless(geo_deps_available, "geopandas/affine/rasterio not available in this env")
class TestGeneratePrompts(unittest.TestCase):

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.video_name = "20230101120000_20230101130000"

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _setup_dirs(self, suffix: str):
        contrail_dir = self.tmp / f"contrail{suffix}"
        contrail_dir.mkdir()
        _write_contrail_parquet(contrail_dir, self.video_name)
        coco_path = _make_synthetic_coco(self.tmp / f"coco{suffix}", self.video_name)
        out_dir = self.tmp / f"prompts{suffix}"
        return contrail_dir, coco_path, out_dir

    def test_creates_output_structure(self):
        from contrailtrack.prompts.writer import generate_prompts
        contrail_dir, coco_path, out_dir = self._setup_dirs("a")
        stats = generate_prompts(
            contrail_dir=contrail_dir,
            coco_annotations=coco_path,
            output_dir=out_dir,
            write_gt_masks=True,
            write_symlinks=False,
        )
        self.assertGreater(stats["videos_processed"], 0)
        self.assertTrue((out_dir / "00001").exists())

    def test_skipped_video_recorded(self):
        from contrailtrack.prompts.writer import generate_prompts
        # Empty contrail dir — all videos should be skipped
        contrail_dir = self.tmp / "empty_contrail"
        contrail_dir.mkdir()
        coco_path = _make_synthetic_coco(self.tmp / "coco_skip", self.video_name)
        out_dir = self.tmp / "prompts_skip"
        stats = generate_prompts(
            contrail_dir=contrail_dir,
            output_dir=out_dir,
            coco_annotations=coco_path,
        )
        self.assertEqual(stats["videos_processed"], 0)
        self.assertGreater(stats["videos_skipped"], 0)

    def test_without_annotations(self):
        """generate_prompts works without coco_annotations (annotation-free batch mode)."""
        from contrailtrack.prompts.writer import generate_prompts
        contrail_dir, _, out_dir = self._setup_dirs("g")
        stats = generate_prompts(
            contrail_dir=contrail_dir,
            output_dir=out_dir,
        )
        self.assertGreater(stats["videos_processed"], 0)
        self.assertTrue((out_dir / "00001").exists())

    def test_no_flight_mapping_json(self):
        """flight_mapping.json must NOT be written — folder name IS the flight ID."""
        from contrailtrack.prompts.writer import generate_prompts
        contrail_dir, coco_path, out_dir = self._setup_dirs("b")
        generate_prompts(
            contrail_dir=contrail_dir,
            coco_annotations=coco_path,
            output_dir=out_dir,
            write_symlinks=False,
        )
        mapping_files = list(out_dir.rglob("flight_mapping.json"))
        self.assertEqual(len(mapping_files), 0, "flight_mapping.json should not be written")

    def test_object_folders_named_by_flight_id(self):
        """Object subfolders must be named by flight_id, not integer indices."""
        from contrailtrack.prompts.writer import generate_prompts
        contrail_dir, coco_path, out_dir = self._setup_dirs("c")
        generate_prompts(
            contrail_dir=contrail_dir,
            coco_annotations=coco_path,
            output_dir=out_dir,
            write_symlinks=False,
        )
        video_dir = out_dir / "00001"
        if video_dir.exists():
            obj_dirs = [d.name for d in video_dir.iterdir() if d.is_dir()]
            for name in obj_dirs:
                self.assertFalse(name.isdigit(),
                    f"Object folder '{name}' should be a flight_id, not an integer")

    def test_prompt_png_values_in_0_255(self):
        from contrailtrack.prompts.writer import generate_prompts
        contrail_dir, coco_path, out_dir = self._setup_dirs("d")
        generate_prompts(
            contrail_dir=contrail_dir,
            coco_annotations=coco_path,
            output_dir=out_dir,
            write_symlinks=False,
        )
        prompt_files = list(out_dir.rglob("*_prompt.png"))
        self.assertGreater(len(prompt_files), 0, "No prompt PNGs written")
        for p in prompt_files:
            arr = np.array(Image.open(p))
            self.assertGreaterEqual(int(arr.min()), 0)
            self.assertLessEqual(int(arr.max()), 255)

    def test_union_covers_all_flights(self):
        """Union mask must be >= any individual flight prompt at every pixel."""
        from contrailtrack.prompts.writer import generate_prompts
        contrail_dir, coco_path, out_dir = self._setup_dirs("e")
        generate_prompts(
            contrail_dir=contrail_dir,
            coco_annotations=coco_path,
            output_dir=out_dir,
            write_symlinks=False,
        )
        for union_file in out_dir.rglob("*_all_prompts_union.png"):
            vid_dir = union_file.parent
            frame_str = union_file.name.replace("_all_prompts_union.png", "")
            per_obj = [
                np.array(Image.open(obj_folder / f"{frame_str}_prompt.png")).astype(float)
                for obj_folder in vid_dir.iterdir()
                if obj_folder.is_dir() and (obj_folder / f"{frame_str}_prompt.png").exists()
            ]
            if not per_obj:
                continue
            union_arr = np.array(Image.open(union_file)).astype(float)
            max_individual = np.max(np.stack(per_obj, axis=0), axis=0)
            self.assertTrue(np.all(union_arr >= max_individual - 1),
                            "Union mask smaller than individual prompt")

    def test_prompt_reader_can_read_output(self):
        """Prompts written by generate_prompts must be readable by read_prompts."""
        from contrailtrack.prompts.writer import generate_prompts
        from contrailtrack.data.prompt_reader import read_prompts, list_objects
        contrail_dir, coco_path, out_dir = self._setup_dirs("f")
        generate_prompts(
            contrail_dir=contrail_dir,
            coco_annotations=coco_path,
            output_dir=out_dir,
            write_symlinks=False,
        )
        objs = list_objects(out_dir, "00001")
        self.assertGreater(len(objs), 0)
        prompts = read_prompts(out_dir, "00001", encoding="ternary")
        self.assertGreater(len(prompts), 0)
        for flight_id, obj_prompts in prompts.items():
            self.assertIsInstance(flight_id, str)
            for frame_name, arr in obj_prompts.items():
                self.assertIsInstance(arr, np.ndarray)
                self.assertEqual(arr.dtype, np.float32)


# ─── generate_prompts_video (annotation-free) ─────────────────────────────────

@unittest.skipUnless(geo_deps_available, "geopandas/affine/rasterio not available in this env")
class TestGeneratePromptsVideo(unittest.TestCase):

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.video_name = "20230101120000_20230101130000"

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _setup(self):
        contrail_df = _make_synthetic_contrail_df(self.video_name)
        images_dir = self.tmp / "frames"
        _make_jpeg_frames_timestamped(images_dir, self.video_name)
        out_dir = self.tmp / "prompts"
        return contrail_df, images_dir, out_dir

    def test_creates_output_structure(self):
        from contrailtrack.prompts.writer import generate_prompts_video
        from contrailtrack.prompts.projection import MiniProjector
        contrail_df, images_dir, out_dir = self._setup()
        stats = generate_prompts_video(
            images_dir=images_dir,
            contrail_df=contrail_df,
            output_dir=out_dir,
            video_id="test_video",
            projector=MiniProjector(),
        )
        self.assertTrue((out_dir / "test_video").exists())
        self.assertGreater(stats["total_objects"], 0)

    def test_object_folders_named_by_flight_id(self):
        """Object subfolders must be named by flight_id, not integer indices."""
        from contrailtrack.prompts.writer import generate_prompts_video
        from contrailtrack.prompts.projection import MiniProjector
        contrail_df, images_dir, out_dir = self._setup()
        generate_prompts_video(
            images_dir=images_dir,
            contrail_df=contrail_df,
            output_dir=out_dir,
            video_id="test_video",
            projector=MiniProjector(),
        )
        video_dir = out_dir / "test_video"
        if video_dir.exists():
            obj_dirs = [d.name for d in video_dir.iterdir() if d.is_dir()]
            for name in obj_dirs:
                self.assertFalse(name.isdigit(),
                    f"Object folder '{name}' should be a flight_id, not an integer")

    def test_no_gt_masks_written(self):
        """generate_prompts_video never writes GT masks."""
        from contrailtrack.prompts.writer import generate_prompts_video
        from contrailtrack.prompts.projection import MiniProjector
        contrail_df, images_dir, out_dir = self._setup()
        generate_prompts_video(
            images_dir=images_dir,
            contrail_df=contrail_df,
            output_dir=out_dir,
            video_id="test_video",
            projector=MiniProjector(),
        )
        self.assertEqual(len(list(out_dir.rglob("*_mask.png"))), 0)

    def test_no_flight_mapping_json(self):
        """flight_mapping.json must NOT be written."""
        from contrailtrack.prompts.writer import generate_prompts_video
        from contrailtrack.prompts.projection import MiniProjector
        contrail_df, images_dir, out_dir = self._setup()
        generate_prompts_video(
            images_dir=images_dir,
            contrail_df=contrail_df,
            output_dir=out_dir,
            video_id="test_video",
            projector=MiniProjector(),
        )
        self.assertEqual(len(list(out_dir.rglob("flight_mapping.json"))), 0)

    def test_prompt_reader_can_read_output(self):
        """Prompts written by generate_prompts_video must be readable by read_prompts."""
        from contrailtrack.prompts.writer import generate_prompts_video
        from contrailtrack.prompts.projection import MiniProjector
        from contrailtrack.data.prompt_reader import read_prompts, list_objects
        contrail_df, images_dir, out_dir = self._setup()
        generate_prompts_video(
            images_dir=images_dir,
            contrail_df=contrail_df,
            output_dir=out_dir,
            video_id="test_video",
            projector=MiniProjector(),
        )
        objs = list_objects(out_dir, "test_video")
        self.assertGreater(len(objs), 0)
        prompts = read_prompts(out_dir, "test_video", encoding="ternary")
        self.assertGreater(len(prompts), 0)

    def test_missing_frames_raises(self):
        from contrailtrack.prompts.writer import generate_prompts_video
        from contrailtrack.prompts.projection import MiniProjector
        contrail_df, _, out_dir = self._setup()
        empty_dir = self.tmp / "empty_frames"
        empty_dir.mkdir()
        with self.assertRaises(FileNotFoundError):
            generate_prompts_video(
                images_dir=empty_dir,
                contrail_df=contrail_df,
                output_dir=out_dir,
                video_id="test_video",
                projector=MiniProjector(),
            )


# ─── integration tests ────────────────────────────────────────────────────────

@unittest.skipUnless(geo_deps_available and real_data_available,
                     "geo deps or real GVCCS data not available")
class TestGeneratePromptsIntegration(unittest.TestCase):

    @unittest.skip("slow: processes all videos (~30min) — run manually only")
    def test_real_data_produces_prompts(self):
        from contrailtrack.prompts.writer import generate_prompts
        tmp = Path(tempfile.mkdtemp())
        self.addCleanup(__import__("shutil").rmtree, tmp, True)
        out_dir = tmp / "real_prompts"
        stats = generate_prompts(
            contrail_dir=REAL_CONTRAIL_DIR,
            coco_annotations=REAL_ANNOTATIONS,
            output_dir=out_dir,
            write_gt_masks=False,
            write_symlinks=False,
        )
        self.assertGreater(stats["videos_processed"], 0)
        self.assertGreater(stats["total_prompt_files_written"], 0)

    @unittest.skip("slow: processes all videos (~30min) — run manually only")
    def test_real_data_readable_by_prompt_reader(self):
        from contrailtrack.prompts.writer import generate_prompts
        from contrailtrack.data.prompt_reader import list_objects
        tmp = Path(tempfile.mkdtemp())
        self.addCleanup(__import__("shutil").rmtree, tmp, True)
        out_dir = tmp / "real_prompts2"
        generate_prompts(
            contrail_dir=REAL_CONTRAIL_DIR,
            coco_annotations=REAL_ANNOTATIONS,
            output_dir=out_dir,
            write_gt_masks=False,
            write_symlinks=False,
        )
        for video_dir in list(out_dir.iterdir())[:3]:
            objs = list_objects(out_dir, video_dir.name)
            self.assertGreater(len(objs), 0, f"No objects for video {video_dir.name}")


if __name__ == "__main__":
    unittest.main()
