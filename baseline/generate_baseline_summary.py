"""Create a concise Markdown report from generated baseline artifacts."""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.reporting import REPORTS_DIR, gallery_statistics
from src.utils import DATASET_DIR, ensure_dir, load_json


def main() -> None:
    directory = ensure_dir(REPORTS_DIR)
    dataset = load_json(DATASET_DIR / "dataset_stats.json", default={}) or {}
    performance = {}
    performance_path = directory / "performance_report.csv"
    if performance_path.exists():
        lines = performance_path.read_text(encoding="utf-8").splitlines()[1:]
        performance = {line.split(",", 1)[0]: line.split(",", 1)[1] for line in lines if "," in line}
    evaluation = load_json(directory / "evaluation_metrics.json", default={}) or {}
    threshold = (directory / "best_threshold.txt").read_text(encoding="utf-8") if (directory / "best_threshold.txt").exists() else "Run evaluate_thresholds.py to generate threshold analysis."
    report = f"""# Face Recognition Baseline Summary

## Project pipeline
Input image, folder, or webcam frame → SCRFD detection → ArcFace embedding → cosine-similarity gallery match → known identity or `UNKNOWN`.

## Dataset statistics
```json
{json.dumps(dataset, indent=2)}
```

## Gallery statistics
```json
{json.dumps(gallery_statistics(), indent=2)}
```

## Performance
```json
{json.dumps(performance, indent=2)}
```

## Evaluation metrics
```json
{json.dumps(evaluation, indent=2)}
```

## Threshold analysis
{threshold}

## Current limitations
- Performance depends on image quality, pose, lighting, gallery coverage, and the chosen threshold.
- This is a pretrained research baseline; it is not a production authentication or surveillance system.
- LFW test images do not represent every real-world demographic or capture condition.

## Future improvements
- Calibrate the threshold on a separate validation set.
- Add liveness detection and consent-aware audit controls.
- Add GPU inference and video-file processing for higher throughput.
- Evaluate with a larger, representative, consented dataset.
"""
    (directory / "baseline_summary.md").write_text(report, encoding="utf-8")


if __name__ == "__main__":
    main()
