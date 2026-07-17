from __future__ import annotations

import logging
import shutil
import tarfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import requests
from sklearn.datasets import fetch_lfw_people
from sklearn.datasets import get_data_home

from .utils import (
    DATASET_DIR,
    GALLERY_DIR,
    KNOWN_TEST_DIR,
    ROOT_DIR,
    TEST_IMAGES_DIR,
    UNKNOWN_TEST_DIR,
    ensure_dir,
    iter_image_files,
    save_json,
    slugify,
)

LOGGER = logging.getLogger(__name__)


@dataclass(slots=True)
class DatasetManifest:
    raw_dataset_dir: Path
    selected_identities: list[dict[str, Any]]
    unknown_identities: list[dict[str, Any]]
    stats: dict[str, Any]


class LFWDatasetManager:
    def __init__(self, dataset_dir: Path | None = None, seed: int = 42) -> None:
        self.dataset_dir = dataset_dir or DATASET_DIR
        self.raw_dir = self.dataset_dir / "raw"
        self.cache_dir = self.dataset_dir / "cache"
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.logger = logging.getLogger(f"{__name__}.LFWDatasetManager")

    def download_dataset(self) -> Path:
        ensure_dir(self.raw_dir)
        try:
            fetch_lfw_people(
                data_home=str(self.cache_dir),
                funneled=True,
                resize=0.5,
                min_faces_per_person=1,
                color=False,
                download_if_missing=True,
            )
        except Exception as exc:
            self.logger.warning("fetch_lfw_people failed, falling back to the official LFW archive: %s", exc)
            self._download_official_archive()

        cached_root = Path(get_data_home(str(self.cache_dir))) / "lfw_home" / "lfw_funneled"
        local_root = self.raw_dir / "lfw_funneled"
        if cached_root.exists():
            if local_root.exists():
                shutil.rmtree(local_root)
            shutil.copytree(cached_root, local_root)
        elif not local_root.exists():
            self._download_official_archive()

        if not local_root.exists():
            raise FileNotFoundError("Unable to populate the LFW dataset directory")
        return local_root

    def _download_official_archive(self) -> None:
        url = "http://vis-www.cs.umass.edu/lfw/lfw.tgz"
        archive_path = self.raw_dir / "lfw.tgz"
        extract_root = self.raw_dir / "lfw_funneled"
        ensure_dir(self.raw_dir)

        if not archive_path.exists():
            response = requests.get(url, stream=True, timeout=120)
            response.raise_for_status()
            with archive_path.open("wb") as handle:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        handle.write(chunk)

        if extract_root.exists():
            shutil.rmtree(extract_root)
        with tarfile.open(archive_path, "r:gz") as archive:
            archive.extractall(self.raw_dir)

    def _scan_identities(self, raw_root: Path) -> pd.DataFrame:
        records: list[dict[str, Any]] = []
        for identity_dir in sorted([path for path in raw_root.iterdir() if path.is_dir()]):
            image_paths = sorted([path for path in identity_dir.iterdir() if path.suffix.lower() in {".jpg", ".jpeg", ".png"}])
            for index, image_path in enumerate(image_paths):
                records.append(
                    {
                        "identity": identity_dir.name,
                        "image_path": str(image_path),
                        "index": index,
                    }
                )
        if not records:
            raise ValueError(f"No images found under {raw_root}")
        return pd.DataFrame.from_records(records)

    def prepare_splits(self) -> DatasetManifest:
        raw_root = self.download_dataset()
        dataframe = self._scan_identities(raw_root)
        counts = dataframe.groupby("identity").size().sort_values(ascending=False)
        eligible = counts[counts >= 20]
        if eligible.empty:
            raise ValueError("No identities with at least 20 images were found in LFW")

        selected_count = min(10, len(eligible))
        if selected_count < 5:
            selected_count = len(eligible)
        selected_identities = list(eligible.index[:selected_count])
        gallery_manifest: list[dict[str, Any]] = []
        unknown_manifest: list[dict[str, Any]] = []

        ensure_dir(GALLERY_DIR)
        ensure_dir(KNOWN_TEST_DIR)
        ensure_dir(UNKNOWN_TEST_DIR)

        for target_identity in selected_identities:
            identity_df = dataframe[dataframe["identity"] == target_identity].sort_values("index").head(20)
            if len(identity_df) < 20:
                continue
            folder_name = slugify(target_identity)
            gallery_dir = ensure_dir(GALLERY_DIR / folder_name)
            known_dir = ensure_dir(KNOWN_TEST_DIR / folder_name)

            rows = identity_df.to_dict("records")
            gallery_rows = rows[:15]
            known_rows = rows[15:20]

            for row in gallery_rows:
                shutil.copy2(row["image_path"], gallery_dir / Path(row["image_path"]).name)
            for row in known_rows:
                shutil.copy2(row["image_path"], known_dir / Path(row["image_path"]).name)

            gallery_manifest.append(
                {
                    "folder_name": folder_name,
                    "original_identity": target_identity,
                    "gallery_images": len(gallery_rows),
                    "known_test_images": len(known_rows),
                }
            )

        non_selected = [identity for identity in eligible.index if identity not in selected_identities]
        unknown_identities = non_selected[:3]
        total_unknown_images = 20
        if unknown_identities:
            base = total_unknown_images // len(unknown_identities)
            remainder = total_unknown_images % len(unknown_identities)
            for offset, target_identity in enumerate(unknown_identities):
                allocate = base + (1 if offset < remainder else 0)
                if allocate <= 0:
                    continue
                identity_df = dataframe[dataframe["identity"] == target_identity].sort_values("index").head(allocate)
                folder_name = slugify(target_identity)
                output_dir = ensure_dir(UNKNOWN_TEST_DIR / folder_name)
                for row in identity_df.to_dict("records"):
                    shutil.copy2(row["image_path"], output_dir / Path(row["image_path"]).name)
                unknown_manifest.append(
                    {
                        "folder_name": folder_name,
                        "original_identity": target_identity,
                        "unknown_images": len(identity_df),
                    }
                )

        stats = {
            "raw_images": int(len(dataframe)),
            "eligible_identities": int(len(eligible)),
            "selected_identities": int(len(gallery_manifest)),
            "unknown_identities": int(len(unknown_manifest)),
            "gallery_images": int(sum(item["gallery_images"] for item in gallery_manifest)),
            "known_test_images": int(sum(item["known_test_images"] for item in gallery_manifest)),
            "unknown_test_images": int(sum(item["unknown_images"] for item in unknown_manifest)),
        }

        manifest = DatasetManifest(
            raw_dataset_dir=raw_root,
            selected_identities=gallery_manifest,
            unknown_identities=unknown_manifest,
            stats=stats,
        )
        save_json(self.dataset_dir / "selection_manifest.json", {
            "raw_dataset_dir": str(raw_root),
            "selected_identities": gallery_manifest,
            "unknown_identities": unknown_manifest,
            "stats": stats,
        })
        save_json(GALLERY_DIR / "identity_map.json", {item["folder_name"]: item["original_identity"] for item in gallery_manifest})
        save_json(TEST_IMAGES_DIR / "unknown_identity_map.json", {item["folder_name"]: item["original_identity"] for item in unknown_manifest})
        save_json(self.dataset_dir / "dataset_stats.json", stats)
        return manifest
