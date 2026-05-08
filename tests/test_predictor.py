"""Tests for contrailtrack.inference.predictor — run_video lazy loading.

All tests use unittest.mock to avoid loading a real SAM2 checkpoint.
The lazy-loading path is exercised by verifying:
  1. prompts_loader is called once per batch with the correct obj_ids slice
  2. Predictions returned by the loader-based path match those of the
     existing dict-based path (same engine calls, same merge logic)
  3. All existing call signatures continue to work unchanged
"""

from __future__ import annotations

import unittest
from unittest.mock import patch, MagicMock, call
from collections import defaultdict

import numpy as np
import torch


_ENGINE = "contrailtrack.inference.predictor.dense_prompt_inference_multi_object"

H, W = 32, 32
T = 4


def _frames():
    return torch.zeros(T, 3, H, W)


def _frame_names():
    return [f"f{i:02d}" for i in range(T)]


def _mask():
    return np.zeros((H, W), dtype=np.float32)


def _engine_returns(obj_ids, frame_names):
    """Return value: every obj active on every frame with a zero mask."""
    return {
        i: {oid: {"mask": _mask(), "score": 0.9} for oid in obj_ids}
        for i, _ in enumerate(frame_names)
    }


class TestRunVideoBackwardsCompat(unittest.TestCase):
    """Existing dict-based API must behave identically to before."""

    def _make_prompts(self, obj_ids, frame_names):
        return {oid: {fn: _mask() for fn in frame_names} for oid in obj_ids}

    def test_no_batch_calls_engine_once(self):
        obj_ids = ["a", "b", "c"]
        prompts = self._make_prompts(obj_ids, _frame_names())

        with patch(_ENGINE) as mock_eng:
            mock_eng.return_value = {}
            from contrailtrack.inference.predictor import run_video
            run_video(
                model=MagicMock(),
                frames=_frames(),
                frame_names=_frame_names(),
                prompts=prompts,
                device="cpu",
            )
        mock_eng.assert_called_once()

    def test_batched_calls_engine_per_batch(self):
        obj_ids = ["a", "b", "c", "d", "e"]
        prompts = self._make_prompts(obj_ids, _frame_names())

        with patch(_ENGINE) as mock_eng:
            mock_eng.return_value = {}
            from contrailtrack.inference.predictor import run_video
            run_video(
                model=MagicMock(),
                frames=_frames(),
                frame_names=_frame_names(),
                prompts=prompts,
                object_batch_size=2,
                device="cpu",
            )
        # ceil(5/2) = 3 batches
        self.assertEqual(mock_eng.call_count, 3)

    def test_batched_merges_predictions(self):
        obj_ids = ["a", "b", "c"]
        prompts = self._make_prompts(obj_ids, _frame_names())

        def engine_side_effect(**kwargs):
            pobj = kwargs["prompts_per_obj"]
            return {0: {oid: {"mask": _mask(), "score": 0.9} for oid in pobj}}

        with patch(_ENGINE, side_effect=engine_side_effect):
            from contrailtrack.inference.predictor import run_video
            result = run_video(
                model=MagicMock(),
                frames=_frames(),
                frame_names=_frame_names(),
                prompts=prompts,
                object_batch_size=2,
                device="cpu",
            )

        # All 3 objs should appear in frame 0
        self.assertIn(0, result)
        for oid in obj_ids:
            self.assertIn(oid, result[0])

    def test_empty_prompts_returns_empty(self):
        with patch(_ENGINE) as mock_eng:
            from contrailtrack.inference.predictor import run_video
            result = run_video(
                model=MagicMock(),
                frames=_frames(),
                frame_names=_frame_names(),
                prompts={},
                device="cpu",
            )
        self.assertEqual(result, {})
        mock_eng.assert_not_called()

    def test_neither_prompts_nor_loader_raises(self):
        from contrailtrack.inference.predictor import run_video
        with self.assertRaises((ValueError, TypeError)):
            run_video(
                model=MagicMock(),
                frames=_frames(),
                frame_names=_frame_names(),
                prompts=None,
                device="cpu",
            )


class TestRunVideoLazyLoading(unittest.TestCase):
    """prompts_loader path: loader called per batch, results identical to dict path."""

    def _make_loader(self, frame_names):
        """Loader that returns non-empty zero masks for any requested obj_ids."""
        calls = []

        def loader(obj_ids):
            calls.append(list(obj_ids))
            return {oid: {fn: _mask() for fn in frame_names} for oid in obj_ids}

        return loader, calls

    def test_loader_called_per_batch(self):
        all_obj_ids = ["a", "b", "c", "d", "e"]
        loader, calls = self._make_loader(_frame_names())

        with patch(_ENGINE) as mock_eng:
            mock_eng.return_value = {}
            from contrailtrack.inference.predictor import run_video
            run_video(
                model=MagicMock(),
                frames=_frames(),
                frame_names=_frame_names(),
                prompts=None,
                all_obj_ids=all_obj_ids,
                prompts_loader=loader,
                object_batch_size=2,
                device="cpu",
            )

        self.assertEqual(len(calls), 3)  # ceil(5/2) = 3
        self.assertEqual(calls[0], ["a", "b"])
        self.assertEqual(calls[1], ["c", "d"])
        self.assertEqual(calls[2], ["e"])

    def test_loader_batch_size_one(self):
        all_obj_ids = ["x", "y", "z"]
        loader, calls = self._make_loader(_frame_names())

        with patch(_ENGINE) as mock_eng:
            mock_eng.return_value = {}
            from contrailtrack.inference.predictor import run_video
            run_video(
                model=MagicMock(),
                frames=_frames(),
                frame_names=_frame_names(),
                prompts=None,
                all_obj_ids=all_obj_ids,
                prompts_loader=loader,
                object_batch_size=1,
                device="cpu",
            )

        self.assertEqual(len(calls), 3)
        self.assertEqual(calls, [["x"], ["y"], ["z"]])

    def test_loader_engine_called_per_batch(self):
        all_obj_ids = ["a", "b", "c"]
        loader, _ = self._make_loader(_frame_names())

        with patch(_ENGINE) as mock_eng:
            mock_eng.return_value = {}
            from contrailtrack.inference.predictor import run_video
            run_video(
                model=MagicMock(),
                frames=_frames(),
                frame_names=_frame_names(),
                prompts=None,
                all_obj_ids=all_obj_ids,
                prompts_loader=loader,
                object_batch_size=2,
                device="cpu",
            )

        # ceil(3/2) = 2 engine calls
        self.assertEqual(mock_eng.call_count, 2)

    def test_loader_predictions_merged(self):
        all_obj_ids = ["a", "b", "c"]
        loader, _ = self._make_loader(_frame_names())

        def engine_side_effect(**kwargs):
            pobj = kwargs["prompts_per_obj"]
            return {0: {oid: {"mask": _mask(), "score": 0.8} for oid in pobj}}

        with patch(_ENGINE, side_effect=engine_side_effect):
            from contrailtrack.inference.predictor import run_video
            result = run_video(
                model=MagicMock(),
                frames=_frames(),
                frame_names=_frame_names(),
                prompts=None,
                all_obj_ids=all_obj_ids,
                prompts_loader=loader,
                object_batch_size=2,
                device="cpu",
            )

        self.assertIn(0, result)
        for oid in all_obj_ids:
            self.assertIn(oid, result[0])

    def test_loader_without_all_obj_ids_raises(self):
        from contrailtrack.inference.predictor import run_video
        with self.assertRaises(ValueError):
            run_video(
                model=MagicMock(),
                frames=_frames(),
                frame_names=_frame_names(),
                prompts=None,
                prompts_loader=lambda ids: {},
                all_obj_ids=None,
                object_batch_size=2,
                device="cpu",
            )

    def test_loader_empty_obj_ids_returns_empty(self):
        with patch(_ENGINE) as mock_eng:
            from contrailtrack.inference.predictor import run_video
            result = run_video(
                model=MagicMock(),
                frames=_frames(),
                frame_names=_frame_names(),
                prompts=None,
                all_obj_ids=[],
                prompts_loader=lambda ids: {},
                object_batch_size=2,
                device="cpu",
            )
        self.assertEqual(result, {})
        mock_eng.assert_not_called()

    def test_loader_loader_returning_empty_batch_skips_engine(self):
        """If loader returns no usable prompts for a batch, engine is not called."""
        all_obj_ids = ["a", "b"]
        # Loader returns empty dict for all objects
        loader = lambda ids: {}

        with patch(_ENGINE) as mock_eng:
            from contrailtrack.inference.predictor import run_video
            run_video(
                model=MagicMock(),
                frames=_frames(),
                frame_names=_frame_names(),
                prompts=None,
                all_obj_ids=all_obj_ids,
                prompts_loader=loader,
                object_batch_size=2,
                device="cpu",
            )

        mock_eng.assert_not_called()

    def test_loader_matches_dict_path_engine_args(self):
        """The indexed prompts passed to the engine are the same regardless of path."""
        fn = _frame_names()[0]
        mask = np.ones((H, W), dtype=np.float32) * 0.5
        prompts_dict = {"obj0": {fn: mask}}

        loader_calls = {}

        def capturing_loader(obj_ids):
            return {oid: {fn: mask} for oid in obj_ids if oid == "obj0"}

        captured_dict = {}
        captured_lazy = {}

        def engine_dict(**kwargs):
            captured_dict.update(kwargs["prompts_per_obj"])
            return {}

        def engine_lazy(**kwargs):
            captured_lazy.update(kwargs["prompts_per_obj"])
            return {}

        from contrailtrack.inference.predictor import run_video

        with patch(_ENGINE, side_effect=engine_dict):
            run_video(
                model=MagicMock(), frames=_frames(), frame_names=_frame_names(),
                prompts=prompts_dict, device="cpu",
            )

        with patch(_ENGINE, side_effect=engine_lazy):
            run_video(
                model=MagicMock(), frames=_frames(), frame_names=_frame_names(),
                prompts=None, all_obj_ids=["obj0"],
                prompts_loader=capturing_loader,
                object_batch_size=1, device="cpu",
            )

        # Same obj_id should be present in both captured dicts
        self.assertIn("obj0", captured_dict)
        self.assertIn("obj0", captured_lazy)
        # Same frame index (0 for first frame)
        self.assertIn(0, captured_dict["obj0"])
        self.assertIn(0, captured_lazy["obj0"])


class TestLoadUnionFrames(unittest.TestCase):

    def setUp(self):
        import tempfile, shutil
        from pathlib import Path
        self.tmp = Path(tempfile.mkdtemp())
        self._shutil = shutil

        vid_dir = self.tmp / "vid"
        vid_dir.mkdir()
        from PIL import Image
        for frame in ["00000", "00001"]:
            arr = np.zeros((16, 16), dtype=np.uint8)
            arr[4:12, 4:12] = 200
            Image.fromarray(arr).save(vid_dir / f"{frame}_all_prompts_union.png")

    def tearDown(self):
        self._shutil.rmtree(self.tmp, ignore_errors=True)

    def test_returns_dict_keyed_by_frame(self):
        from contrailtrack.data.prompt_reader import load_union_frames
        unions = load_union_frames(self.tmp, "vid")
        self.assertIn("00000", unions)
        self.assertIn("00001", unions)

    def test_values_are_float32_arrays(self):
        from contrailtrack.data.prompt_reader import load_union_frames
        unions = load_union_frames(self.tmp, "vid")
        arr = unions["00000"]
        self.assertEqual(arr.dtype, np.float32)
        self.assertEqual(arr.shape, (16, 16))

    def test_values_normalized_0_to_1(self):
        from contrailtrack.data.prompt_reader import load_union_frames
        unions = load_union_frames(self.tmp, "vid")
        arr = unions["00000"]
        self.assertGreaterEqual(float(arr.min()), 0.0)
        self.assertLessEqual(float(arr.max()), 1.0)

    def test_missing_video_returns_empty(self):
        from contrailtrack.data.prompt_reader import load_union_frames
        unions = load_union_frames(self.tmp, "no_such_vid")
        self.assertEqual(unions, {})


def _make_prompt_dir(tmp):
    """Minimal per_object_data/ structure: 2 objects, 2 frames."""
    from pathlib import Path
    from PIL import Image
    tmp = Path(tmp)
    vid_dir = tmp / "00001"
    for obj_folder, frames in [("0", ["00000", "00001"]), ("1", ["00000"])]:
        obj_dir = vid_dir / obj_folder
        obj_dir.mkdir(parents=True, exist_ok=True)
        for frame in frames:
            arr = np.zeros((64, 64), dtype=np.uint8)
            arr[10:30, 10:30] = 200
            Image.fromarray(arr).save(obj_dir / f"{frame}_prompt.png")
    for frame in ["00000", "00001"]:
        union = np.zeros((64, 64), dtype=np.uint8)
        union[5:35, 5:35] = 180
        Image.fromarray(union).save(vid_dir / f"{frame}_all_prompts_union.png")
    return tmp


class TestReadPromptsObjIdsFilter(unittest.TestCase):

    def setUp(self):
        import tempfile, shutil
        from pathlib import Path
        self.tmp = Path(tempfile.mkdtemp())
        self._shutil = shutil
        _make_prompt_dir(self.tmp)

    def tearDown(self):
        self._shutil.rmtree(self.tmp, ignore_errors=True)

    def test_obj_ids_none_loads_all(self):
        from contrailtrack.data.prompt_reader import read_prompts
        result = read_prompts(self.tmp, "00001", encoding="binary", obj_ids=None)
        self.assertIn("0", result)
        self.assertIn("1", result)

    def test_obj_ids_filter_single(self):
        from contrailtrack.data.prompt_reader import read_prompts
        result = read_prompts(self.tmp, "00001", encoding="binary", obj_ids=["0"])
        self.assertIn("0", result)
        self.assertNotIn("1", result)

    def test_obj_ids_filter_subset(self):
        from contrailtrack.data.prompt_reader import read_prompts
        result = read_prompts(self.tmp, "00001", encoding="binary", obj_ids=["1"])
        self.assertNotIn("0", result)
        self.assertIn("1", result)

    def test_obj_ids_empty_list_returns_empty(self):
        from contrailtrack.data.prompt_reader import read_prompts
        result = read_prompts(self.tmp, "00001", encoding="binary", obj_ids=[])
        self.assertEqual(result, {})

    def test_obj_ids_nonexistent_returns_empty(self):
        from contrailtrack.data.prompt_reader import read_prompts
        result = read_prompts(self.tmp, "00001", encoding="binary", obj_ids=["no_such"])
        self.assertEqual(result, {})

    def test_obj_ids_filter_ternary(self):
        from contrailtrack.data.prompt_reader import read_prompts
        result = read_prompts(self.tmp, "00001", encoding="ternary", obj_ids=["0"])
        self.assertIn("0", result)
        self.assertNotIn("1", result)


class TestReadPromptsPreloadedUnions(unittest.TestCase):

    def setUp(self):
        import tempfile, shutil
        from pathlib import Path
        self.tmp = Path(tempfile.mkdtemp())
        self._shutil = shutil
        _make_prompt_dir(self.tmp)

    def tearDown(self):
        self._shutil.rmtree(self.tmp, ignore_errors=True)

    def test_preloaded_unions_matches_normal(self):
        from contrailtrack.data.prompt_reader import read_prompts, load_union_frames
        normal = read_prompts(self.tmp, "00001", encoding="ternary")
        unions = load_union_frames(self.tmp, "00001")
        with_preloaded = read_prompts(self.tmp, "00001", encoding="ternary",
                                      union_per_frame=unions)
        for obj_id in normal:
            self.assertIn(obj_id, with_preloaded)
            for frame in normal[obj_id]:
                np.testing.assert_array_equal(
                    normal[obj_id][frame], with_preloaded[obj_id][frame],
                    err_msg=f"Mismatch for obj={obj_id} frame={frame}"
                )

    def test_preloaded_unions_with_filter(self):
        from contrailtrack.data.prompt_reader import read_prompts, load_union_frames
        unions = load_union_frames(self.tmp, "00001")
        result = read_prompts(self.tmp, "00001", encoding="ternary",
                              obj_ids=["0"], union_per_frame=unions)
        self.assertIn("0", result)
        self.assertNotIn("1", result)


if __name__ == "__main__":
    unittest.main()
