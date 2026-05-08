"""Count spatial teleportation glitches in prediction JSONs.

A glitch is defined as a centroid jump > JUMP_THRESHOLD pixels between
consecutive detections of the same object in the same video.

Usage:
    python scripts/count_glitches.py outputs/ternary_5_rerun_remapped/
    python scripts/count_glitches.py outputs/ternary_5_negprompt/
    python scripts/count_glitches.py outputs/dir1/ outputs/dir2/  # compare two runs
"""

import sys
import json
import math
from pathlib import Path
from collections import defaultdict

JUMP_THRESHOLD = 150  # pixels


def bbox_centroid(bbox):
    x, y, w, h = bbox
    return x + w / 2, y + h / 2


def count_glitches_in_file(json_path):
    with open(json_path) as f:
        data = json.load(f)

    # Group by flight_id (object)
    by_object = defaultdict(list)
    for ann in data["annotations"]:
        by_object[ann["flight_id"]].append(ann)

    glitch_count = 0
    glitch_objects = set()

    for flight_id, anns in by_object.items():
        anns_sorted = sorted(anns, key=lambda a: a["frame_idx"])
        prev_cx, prev_cy = None, None
        for ann in anns_sorted:
            cx, cy = bbox_centroid(ann["bbox"])
            if prev_cx is not None:
                dist = math.sqrt((cx - prev_cx) ** 2 + (cy - prev_cy) ** 2)
                if dist > JUMP_THRESHOLD:
                    glitch_count += 1
                    glitch_objects.add(flight_id)
            prev_cx, prev_cy = cx, cy

    return glitch_count, len(glitch_objects), len(by_object)


def count_glitches_in_dir(pred_dir):
    pred_dir = Path(pred_dir)
    json_files = sorted(pred_dir.glob("*.json"))
    if not json_files:
        print(f"  No JSON files found in {pred_dir}")
        return

    total_glitches = 0
    total_glitch_objects = 0
    total_objects = 0

    for jf in json_files:
        glitches, glitch_objs, n_objs = count_glitches_in_file(jf)
        total_glitches += glitches
        total_glitch_objects += glitch_objs
        total_objects += n_objs
        if glitches > 0:
            print(f"  {jf.stem}: {glitches} glitches across {glitch_objs}/{n_objs} objects")

    print(f"\n  TOTAL: {total_glitches} glitches across {total_glitch_objects}/{total_objects} objects "
          f"in {len(json_files)} videos")
    return total_glitches, total_glitch_objects, total_objects, len(json_files)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    dirs = sys.argv[1:]
    results = {}
    for d in dirs:
        print(f"\n=== {d} ===")
        r = count_glitches_in_dir(d)
        if r:
            results[d] = r

    if len(results) == 2:
        dirs_list = list(results.keys())
        g1 = results[dirs_list[0]][0]
        g2 = results[dirs_list[1]][0]
        delta = g2 - g1
        sign = "+" if delta >= 0 else ""
        print(f"\n=== Comparison ===")
        print(f"  {Path(dirs_list[0]).name}: {g1} glitches")
        print(f"  {Path(dirs_list[1]).name}: {g2} glitches")
        print(f"  Delta: {sign}{delta} ({sign}{100*delta/g1:.1f}%)" if g1 > 0 else "")
