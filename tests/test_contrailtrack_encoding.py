"""Tests for contrailtrack.prompts.encoding — prompt encoding functions."""

import unittest
import numpy as np


class TestEncodeBinary(unittest.TestCase):

    def _mask(self):
        m = np.zeros((32, 32), dtype=np.float32)
        m[5:15, 5:15] = 0.7
        return m

    def test_output_is_binary(self):
        from contrailtrack.prompts.encoding import encode_binary
        result = encode_binary(self._mask())
        unique = set(result.flatten())
        self.assertTrue(unique.issubset({0.0, 1.0}))

    def test_positive_pixels_become_one(self):
        from contrailtrack.prompts.encoding import encode_binary
        m = self._mask()
        result = encode_binary(m)
        self.assertTrue(np.all(result[5:15, 5:15] == 1.0))

    def test_zero_pixels_remain_zero(self):
        from contrailtrack.prompts.encoding import encode_binary
        m = self._mask()
        result = encode_binary(m)
        self.assertTrue(np.all(result[:5, :5] == 0.0))

    def test_output_dtype_float32(self):
        from contrailtrack.prompts.encoding import encode_binary
        result = encode_binary(self._mask())
        self.assertEqual(result.dtype, np.float32)


class TestEncodeAgeWeighted(unittest.TestCase):

    def test_clips_to_0_1(self):
        from contrailtrack.prompts.encoding import encode_age_weighted
        m = np.array([[-0.1, 0.5, 1.5]], dtype=np.float32)
        result = encode_age_weighted(m)
        self.assertGreaterEqual(result.min(), 0.0)
        self.assertLessEqual(result.max(), 1.0)

    def test_passes_through_valid_values(self):
        from contrailtrack.prompts.encoding import encode_age_weighted
        m = np.array([[0.0, 0.3, 0.7, 1.0]], dtype=np.float32)
        result = encode_age_weighted(m)
        np.testing.assert_array_almost_equal(result, m)


class TestEncodeTernary(unittest.TestCase):

    def _own_union(self, h=32, w=32):
        own = np.zeros((h, w), dtype=np.float32)
        own[10:20, 10:20] = 0.8

        union = np.zeros((h, w), dtype=np.float32)
        union[10:20, 10:20] = 0.8  # overlaps with own
        union[5:10, 5:10] = 0.6   # additional area (other flights)
        return own, union

    def test_positive_region_from_own(self):
        from contrailtrack.prompts.encoding import encode_ternary
        own, union = self._own_union()
        result = encode_ternary(own, union)
        self.assertTrue(np.all(result[10:20, 10:20] > 0))

    def test_negative_region_from_union(self):
        from contrailtrack.prompts.encoding import encode_ternary
        own, union = self._own_union()
        result = encode_ternary(own, union)
        # union-only region (5:10, 5:10) should be negative
        self.assertTrue(np.all(result[5:10, 5:10] < 0))

    def test_zero_where_no_flights(self):
        from contrailtrack.prompts.encoding import encode_ternary
        own, union = self._own_union()
        result = encode_ternary(own, union)
        # Region with neither own nor union should be 0
        self.assertEqual(float(result[0, 0]), 0.0)

    def test_formula_matches_where_statement(self):
        """Formula: np.where(own > 0, own, -union)."""
        from contrailtrack.prompts.encoding import encode_ternary
        own = np.array([[0.5, 0.0, 0.0]], dtype=np.float32)
        union = np.array([[0.3, 0.4, 0.0]], dtype=np.float32)
        expected = np.array([[0.5, -0.4, 0.0]], dtype=np.float32)
        result = encode_ternary(own, union)
        np.testing.assert_array_almost_equal(result, expected)

    def test_output_dtype_float32(self):
        from contrailtrack.prompts.encoding import encode_ternary
        own, union = self._own_union()
        result = encode_ternary(own, union)
        self.assertEqual(result.dtype, np.float32)

    def test_ternary_consistent_with_prompt_reader(self):
        """Ternary formula in encoding.py must match what read_prompts applies."""
        import tempfile
        from pathlib import Path
        from PIL import Image

        # Build synthetic prompt dir
        tmp = Path(tempfile.mkdtemp())
        self.addCleanup(__import__("shutil").rmtree, str(tmp), True)
        vid_dir = tmp / "00001"
        obj_dir = vid_dir / "0"
        obj_dir.mkdir(parents=True)

        own_arr = np.zeros((64, 64), dtype=np.float32)
        own_arr[10:20, 10:20] = 0.8
        union_arr = np.zeros((64, 64), dtype=np.float32)
        union_arr[5:25, 5:25] = 0.5

        Image.fromarray((own_arr * 255).astype(np.uint8)).save(obj_dir / "00000_prompt.png")
        Image.fromarray((union_arr * 255).astype(np.uint8)).save(vid_dir / "00000_all_prompts_union.png")

        from contrailtrack.data.prompt_reader import read_prompts
        from contrailtrack.prompts.encoding import encode_ternary

        prompts = read_prompts(tmp, "00001", encoding="ternary")
        # Reconstruct using encoding.py
        own_loaded = np.array(Image.open(obj_dir / "00000_prompt.png")).astype(np.float32) / 255.0
        union_loaded = np.array(Image.open(vid_dir / "00000_all_prompts_union.png")).astype(np.float32) / 255.0
        expected = encode_ternary(own_loaded, union_loaded)

        np.testing.assert_array_almost_equal(prompts[1]["00000"], expected, decimal=5)


if __name__ == "__main__":
    unittest.main()
