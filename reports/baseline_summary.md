# Face Recognition Baseline Summary

## Project pipeline
Input image, folder, or webcam frame → SCRFD detection → ArcFace embedding → cosine-similarity gallery match → known identity or `UNKNOWN`.

## Dataset statistics
```json
{
  "raw_images": 13233,
  "eligible_identities": 62,
  "selected_identities": 10,
  "unknown_identities": 3,
  "gallery_images": 150,
  "known_test_images": 50,
  "unknown_test_images": 20
}
```

## Gallery statistics
```json
{
  "number_of_identities": 10,
  "images_per_identity": {
    "ariel_sharon": 15,
    "colin_powell": 15,
    "donald_rumsfeld": 15,
    "george_w_bush": 15,
    "gerhard_schroeder": 15,
    "hugo_chavez": 15,
    "jean_chretien": 15,
    "john_ashcroft": 15,
    "junichiro_koizumi": 15,
    "tony_blair": 15
  },
  "embedding_dimension": 512,
  "total_gallery_images": 150
}
```

## Performance
```json
{
  "average_detection_time": "1.233496",
  "average_recognition_time": "0.000077",
  "average_processing_time": "1.233575",
  "fps": "0.810652",
  "average_similarity": "0.539786"
}
```

## Evaluation metrics
```json
{
  "accuracy": 1.0,
  "precision": 1.0,
  "recall": 1.0,
  "f1_score": 1.0,
  "false_positive_rate": 0.0,
  "false_negative_rate": 0.0
}
```

## Threshold analysis
Best threshold: 0.20
F1 score: 1.0000
Accuracy: 1.0000


## Current limitations
- Performance depends on image quality, pose, lighting, gallery coverage, and the chosen threshold.
- This is a pretrained research baseline; it is not a production authentication or surveillance system.
- LFW test images do not represent every real-world demographic or capture condition.

## Future improvements
- Calibrate the threshold on a separate validation set.
- Add liveness detection and consent-aware audit controls.
- Add GPU inference and video-file processing for higher throughput.
- Evaluate with a larger, representative, consented dataset.
