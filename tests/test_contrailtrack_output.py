"""Tests for contrailtrack.output.coco — RLE encoding and JSON export."""

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np


class TestEncodeRle(unittest.TestCase):

    def _solid_mask(self, h=64, w=64):
        mask = np.zeros((h, w), dtype=bool)
        mask[10:30, 10:50] = True
        return mask

    def test_encode_rle_returns_dict_with_size_and_counts(self):
        from contrailtrack.output.coco import encode_rle
        rle = encode_rle(self._solid_mask())
        self.assertIn("size", rle)
        self.assertIn("counts", rle)

    def test_encode_rle_size_matches_mask(self):
        from contrailtrack.output.coco import encode_rle
        mask = self._solid_mask(32, 48)
        rle = encode_rle(mask)
        self.assertEqual(rle["size"], [32, 48])

    def test_encode_rle_counts_is_string(self):
        from contrailtrack.output.coco import encode_rle
        rle = encode_rle(self._solid_mask())
        self.assertIsInstance(rle["counts"], str)

    def test_encode_rle_roundtrip(self):
        """Encode then decode should reproduce the original mask."""
        from contrailtrack.output.coco import encode_rle
        from pycocotools import mask as mask_utils
        mask = self._solid_mask()
        rle = encode_rle(mask)
        # Decode: counts must be bytes for pycocotools
        rle_bytes = {"size": rle["size"], "counts": rle["counts"].encode("utf-8")}
        decoded = mask_utils.decode(rle_bytes).astype(bool)
        np.testing.assert_array_equal(mask, decoded)

    def test_encode_rle_empty_mask(self):
        from contrailtrack.output.coco import encode_rle
        mask = np.zeros((64, 64), dtype=bool)
        rle = encode_rle(mask)
        self.assertIsInstance(rle["counts"], str)

    def test_encode_rle_uint8_input(self):
        """Should accept uint8 mask (not just bool)."""
        from contrailtrack.output.coco import encode_rle
        mask = np.zeros((32, 32), dtype=np.uint8)
        mask[5:10, 5:10] = 1
        rle = encode_rle(mask)
        self.assertIn("counts", rle)


class TestExportCocoJson(unittest.TestCase):

    def _make_predictions(self, h=64, w=64):
        """Build a minimal predictions dict: 2 frames, 2 objects."""
        mask1 = np.zeros((h, w), dtype=bool)
        mask1[10:20, 10:20] = True
        mask2 = np.zeros((h, w), dtype=bool)
        mask2[30:40, 30:40] = True
        return {
            0: {1: {"mask": mask1, "score": 0.9}},
            1: {1: {"mask": mask2, "score": 0.7}, 2: {"mask": mask1, "score": 0.5}},
        }

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.h, self.w = 64, 64
        self.predictions = self._make_predictions(self.h, self.w)
        self.frame_names = ["00000", "00001"]

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_export_creates_file(self):
        from contrailtrack.output.coco import export_coco_json
        out = self.tmp / "test.json"
        result = export_coco_json(
            self.predictions, "00001", self.frame_names, self.h, self.w, out
        )
        self.assertTrue(result.exists())

    def test_export_returns_path(self):
        from contrailtrack.output.coco import export_coco_json
        out = self.tmp / "test.json"
        result = export_coco_json(
            self.predictions, "00001", self.frame_names, self.h, self.w, out
        )
        self.assertIsInstance(result, Path)

    def test_export_json_structure(self):
        from contrailtrack.output.coco import export_coco_json
        out = self.tmp / "test.json"
        export_coco_json(
            self.predictions, "00001", self.frame_names, self.h, self.w, out
        )
        with open(out) as f:
            data = json.load(f)
        self.assertIn("info", data)
        self.assertIn("video", data)
        self.assertIn("annotations", data)

    def test_export_annotation_count(self):
        """3 non-empty annotations (frame0→obj1, frame1→obj1, frame1→obj2)."""
        from contrailtrack.output.coco import export_coco_json
        out = self.tmp / "test.json"
        export_coco_json(
            self.predictions, "00001", self.frame_names, self.h, self.w, out
        )
        with open(out) as f:
            data = json.load(f)
        self.assertEqual(len(data["annotations"]), 3)

    def test_export_annotation_fields(self):
        from contrailtrack.output.coco import export_coco_json
        out = self.tmp / "test.json"
        export_coco_json(
            self.predictions, "00001", self.frame_names, self.h, self.w, out
        )
        with open(out) as f:
            data = json.load(f)
        ann = data["annotations"][0]
        for key in ("id", "flight_id", "frame_idx", "frame_name", "segmentation", "area", "score"):
            self.assertIn(key, ann)

    def test_export_flight_id_matches_internal(self):
        """flight_id in JSON must match the obj_id keys from predictions."""
        from contrailtrack.output.coco import export_coco_json
        out = self.tmp / "test.json"
        export_coco_json(
            self.predictions, "00001", self.frame_names, self.h, self.w, out
        )
        with open(out) as f:
            data = json.load(f)
        flight_ids = {a["flight_id"] for a in data["annotations"]}
        self.assertEqual(flight_ids, {"1", "2"})

    def test_export_empty_mask_skipped(self):
        """Annotations with area=0 must not appear in output."""
        from contrailtrack.output.coco import export_coco_json
        preds = {
            0: {1: {"mask": np.zeros((64, 64), dtype=bool), "score": 0.9}},
        }
        out = self.tmp / "empty_mask.json"
        export_coco_json(preds, "00001", ["00000"], 64, 64, out)
        with open(out) as f:
            data = json.load(f)
        self.assertEqual(len(data["annotations"]), 0)

    def test_export_creates_parent_dirs(self):
        from contrailtrack.output.coco import export_coco_json
        out = self.tmp / "nested" / "deep" / "test.json"
        export_coco_json(
            self.predictions, "00001", self.frame_names, self.h, self.w, out
        )
        self.assertTrue(out.exists())

    def test_export_metadata_included_in_info(self):
        from contrailtrack.output.coco import export_coco_json
        out = self.tmp / "meta.json"
        export_coco_json(
            self.predictions, "00001", self.frame_names, self.h, self.w, out,
            metadata={"checkpoint": "ternary_5/checkpoint.pt"}
        )
        with open(out) as f:
            data = json.load(f)
        self.assertEqual(data["info"]["checkpoint"], "ternary_5/checkpoint.pt")


if __name__ == "__main__":
    unittest.main()
