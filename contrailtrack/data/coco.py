"""COCO dataset utilities with video support.

COCOWithVideo and COCOWithPolygons are adapted from the trailvision library.
"""

from __future__ import annotations

import copy
from typing import Dict, List

from pycocotools.coco import COCO


class COCOWithPolygons(COCO):
    def __init__(self, annotation_file: str | None = None, split_polygons: bool = False):
        super().__init__(annotation_file)
        self.split_polygons = split_polygons
        if self.split_polygons:
            self._split_annotations()

    def _split_annotations(self):
        new_anns = []
        ann_id = 1
        for ann in self.dataset["annotations"]:
            for poly in ann["segmentation"]:
                new_ann = copy.deepcopy(ann)
                new_ann["segmentation"] = [poly]
                new_ann["instance_id"] = ann["id"]
                new_ann["id"] = ann_id
                ann_id += 1
                new_anns.append(new_ann)
        self.dataset["annotations"] = new_anns
        self.createIndex()


class COCOWithVideo(COCOWithPolygons):
    """COCO dataset class extended with video-level indexing.

    Supports getVideoIds(), loadVideos(), and getImgIds(videoIds=...).
    """

    def __init__(self, annotation_file=None, split_polygons=False):
        super().__init__(annotation_file, split_polygons=split_polygons)
        if "videos" in self.dataset:
            self.createIndexVideos()

    def createIndexVideos(self) -> None:
        self.vids: Dict[int, dict] = {}
        self.videoToImgs: Dict[int, List[dict]] = {}

        for video in self.dataset["videos"]:
            self.vids[video["id"]] = video
            self.videoToImgs[video["id"]] = []

        for img in self.dataset["images"]:
            vid_id = img.get("video_id")
            if vid_id is not None:
                self.videoToImgs[vid_id].append(img)

    def getVideoIds(self) -> List[int]:
        return sorted(self.vids.keys())

    def loadVideos(self, ids: List[int] | None = None) -> List[dict]:
        return [self.vids[vid] for vid in (ids or [])]

    def getImgIds(
        self,
        videoIds: List[int] | None = None,
        imgIds: List[int] | None = None,
        catIds: List[int] | None = None,
    ) -> List[int]:
        img_ids = set(super().getImgIds(imgIds=imgIds or [], catIds=catIds or []))
        if videoIds:
            vid_img_ids = set()
            for vid in videoIds:
                vid_img_ids.update(img["id"] for img in self.videoToImgs.get(vid, []))
            img_ids &= vid_img_ids
        return sorted(img_ids)
