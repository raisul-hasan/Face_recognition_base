# Trainable Face Recognition Model

## Face Recognition Overview

Modern face recognition combines face detection, landmark alignment, feature extraction, and an identity decision. SCRFD first locates each face and returns a bounding box, facial landmarks, and a detection score. Landmark alignment presents a consistently positioned face to the recognition network. The model then represents the face as an embedding: a compact numerical vector in which images of the same person tend to be close and images of different people tend to be separated.

This project retains detection confidence separately from identity confidence. A detector may correctly find a face even when that face has not been enrolled in the recognition system. This distinction is important for transparent evaluation and for open-set recognition, where unfamiliar people must not be forced into an enrolled class.

## ArcFace Feature Extractor

Pretrained ArcFace from InsightFace `buffalo_l` is used exclusively as the feature extractor. Every gallery image is processed by SCRFD, aligned from facial landmarks, and converted into a normalized ArcFace embedding. These samples are saved as `training/embeddings.npy`, with matching labels in `training/labels.npy`.

ArcFace is an appropriate fixed backbone because its angular-margin training objective produces discriminative facial representations. Training ArcFace from scratch would require an extremely large curated face dataset, considerable GPU computation, and careful treatment of privacy and demographic bias. The LFW-scale gallery is suitable for a lightweight classifier but is not sufficient to train a reliable deep face backbone. A fixed pretrained extractor is therefore a practical transfer-learning solution.

## Trained Classification Model

The trainable component is a scikit-learn classifier trained on ArcFace embeddings. The data are split using an 80/20 stratified partition, and a standard scaler is fitted using training samples only. Linear SVM, RBF SVM, logistic regression, and K-nearest neighbours are evaluated automatically. For each candidate, the system saves accuracy, weighted precision, recall, F1 score, classification report, confusion matrix, and one-versus-rest ROC curve where applicable.

SVM is well suited to high-dimensional face embeddings because it estimates margin-based decision boundaries. A linear SVM provides a strong efficient baseline, while an RBF SVM can model non-linear class boundaries. The final choice is empirical rather than assumed: the candidate with the highest held-out accuracy is saved. The classifier, label encoder, scaler, and threshold metadata are persisted under `models/` with joblib, so recognition can run without retraining.

## Recognition Process

For a new image, SCRFD detects the face, ArcFace extracts its embedding, the saved scaler transforms that vector, and the saved classifier predicts a class probability. The label encoder converts the selected class back to a person name. The maximum probability is compared with a calibrated confidence threshold. If it passes, the person identity and confidence are displayed; otherwise the output is **NOT RECOGNIZED**.

Thresholds from 0.50 to 0.90 are evaluated using separate known and unknown images. The selected operating point is reported with recognition accuracy, false acceptance rate, false rejection rate, and an open-set confusion matrix. The annotated image shows the bounding box, classifier confidence, detection confidence, and recognition time. This thresholded decision prevents the closed-set classifier from claiming that every new face belongs to a known identity.
