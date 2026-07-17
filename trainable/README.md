# Trainable Component

This folder groups the machine-learning extension. ArcFace remains fixed as a
pretrained feature extractor; the scikit-learn classifier is the trained model.

Run these commands from the project root:

```powershell
python trainable/train_model.py
python trainable/load_model.py --input test_images/known/tony_blair/Tony_Blair_0016.jpg
```

Training outputs are deliberately kept at the project root:

- `training/` — embeddings and train/test split
- `models/` — saved classifier, scaler, encoder, and threshold
- `reports/` — model metrics and plots

The original root scripts remain available for backward compatibility.
