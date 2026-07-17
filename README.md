# Trainable Face Recognition System

This project preserves the original pretrained baseline and adds a trainable scikit-learn identity classifier. SCRFD detects/aligned faces and pretrained ArcFace (`buffalo_l`) extracts embeddings; ArcFace is never trained from scratch.

It uses:

- InsightFace SCRFD for detection
- InsightFace ArcFace embeddings for recognition
- Cosine similarity for identity matching
- OpenCV for visualization and webcam inference

## Features

- Automatic LFW dataset download
- Automatic gallery selection from identities with at least 20 images
- Automatic train/test split for known faces
- Automatic unknown-face test set construction from identities not used in the gallery
- Multi-face recognition on images
- Live webcam recognition
- Annotated output image export

## Installation

Create a Python 3.10 environment and install dependencies:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

If you are using a system interpreter, make sure the required packages are available before running the scripts.

## Folder Structure

The two project parts are grouped for easy navigation:

- `baseline/` contains entry points for dataset/gallery preparation and the original recognition workflows.
- `trainable/` contains entry points for classifier training and loading the saved classifier.

The root-level scripts are intentionally retained so earlier commands continue to work.

The project creates the following structure automatically:

```text
FaceRecognition_Baseline/
├── dataset/
├── gallery/
├── embeddings/
├── outputs/
├── test_images/
│   ├── known/
│   └── unknown/
├── notebooks/
├── src/
├── build_gallery.py
├── recognize_image.py
├── webcam.py
├── download_dataset.py
├── requirements.txt
└── README.md
```

## Dataset Download

Run:

```bash
python baseline/download_dataset.py
```

The pipeline first tries `sklearn.datasets.fetch_lfw_people()` with the official LFW dataset cache. If that fails, it falls back to the official LFW archive.

The raw dataset is stored under `dataset/raw/lfw_funneled/`.

## Gallery Generation

Run:

```bash
python baseline/build_gallery.py --prepare-dataset
```

This will:

1. Download and organize the dataset
2. Select 5 to 10 identities with at least 20 images
3. Copy 15 images per selected identity into `gallery/`
4. Copy the remaining 5 images per selected identity into `test_images/known/`
5. Select identities not used in the gallery and place unknown samples in `test_images/unknown/`
6. Build averaged gallery embeddings in `embeddings/`

The selected folder names are mapped back to their original LFW identities in `gallery/identity_map.json`.

## Image Recognition

## Training the Classifier

After preparing the gallery, train the classifier with:

```bash
python trainable/train_model.py
```

The command produces one aligned ArcFace embedding per gallery image in `training/embeddings.npy` with labels in `training/labels.npy`. It uses a stratified 80/20 split, compares Linear SVM, RBF SVM, Logistic Regression, and KNN, then automatically selects the highest-accuracy candidate. Per-classifier accuracy, precision, recall, F1, confusion matrices, classification reports, ROC plots, and comparison plots are stored in `reports/`.

The selected model is saved in `models/` as `best_classifier.joblib`, `label_encoder.joblib`, `scaler.joblib`, and `threshold.json`. Calibration evaluates confidence thresholds 0.50 through 0.90 against known and unknown images, including recognition accuracy, false-acceptance rate, false-rejection rate, and an open-set confusion matrix.

The inference pipeline is SCRFD detection → aligned ArcFace embedding → scaler → trained classifier → calibrated probability threshold. Below-threshold faces display `NOT RECOGNIZED`; accepted faces display identity, classifier confidence, detection confidence, and recognition time.

Load the saved model for a simple prediction with:

```bash
python trainable/load_model.py --input test_images/known/tony_blair/Tony_Blair_0016.jpg
```

To add an identity, put images in `gallery/<identity>/`, then run `python baseline/build_gallery.py` and `python trainable/train_model.py`.

Run recognition on a single image or directory:

```bash
python baseline/recognize_image.py --input test_images --threshold 0.35
```

Annotated output images are saved in `outputs/`.

## Research Evaluation and Reporting

Process the complete test-image folder and save annotated results, crops, CSV logs, and performance metrics:

```bash
python baseline/recognize_folder.py
```

Evaluate the required similarity thresholds and generate the classification report, confusion matrix, and threshold graph:

```bash
python baseline/evaluate_thresholds.py
python baseline/generate_baseline_summary.py
```

Generated research artifacts are saved under `reports/`, recognition events under `logs/recognition_log.csv`, face crops under `outputs/cropped_faces/`, and gallery statistics in `gallery_report.json`.

Project settings, including the similarity threshold and visualization styling, are stored in `config.yaml`.

## Threshold Tuning

The default cosine-similarity threshold is `0.35`.

- Increase the threshold to reduce false positives
- Decrease the threshold to recognize more borderline matches

The best value depends on how similar your gallery faces are and how strict you want unknown handling to be.

## Webcam Recognition

Run live webcam inference:

```bash
python baseline/webcam.py
```

Press `Q` to quit.

For validation in environments without a camera, the script exits gracefully if the webcam cannot be opened.

## Full Pipeline

Run the complete automation flow:

```bash
python baseline/run_pipeline.py --webcam-frames 10
```

This executes dataset preparation, gallery embedding generation, known-image recognition, unknown-image recognition, and a short webcam sanity pass.

## Notebook

Open and run:

`notebooks/Baseline_Demo.ipynb`

The notebook demonstrates:

1. Dataset download
2. Gallery preparation
3. Embedding generation
4. Face detection
5. Face recognition
6. Unknown-face handling
7. Saving outputs
8. Final visual results

## Adding New Identities

To add identities to the gallery, place their images under `gallery/<identity_name>/`, then rerun:

```bash
python baseline/build_gallery.py
python trainable/train_model.py
```

The script will rebuild the averaged embeddings automatically.

## Notes

- No neural network training is performed.
- The project is built entirely from pretrained InsightFace models.
- If you run into dependency issues on a newer Python version, use Python 3.10 for the best compatibility.
