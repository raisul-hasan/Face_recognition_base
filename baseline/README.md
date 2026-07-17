# Baseline Component

This folder provides grouped entry points for the original pretrained-only
baseline. Its pipeline is SCRFD detection → ArcFace embedding → cosine
similarity against averaged gallery embeddings.

Run these commands from the project root:

```powershell
python baseline/build_gallery.py
python baseline/recognize_image.py --input test_images
python baseline/recognize_folder.py
python baseline/webcam.py
```

The original root scripts remain available for backward compatibility. The
baseline is automatically used as a fallback when no trained classifier exists.
