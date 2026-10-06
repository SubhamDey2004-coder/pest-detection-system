# AI-Powered Pest & Crop Disease Detection System

An image-based agricultural AI system that classifies crop disease and pest-related damage and exposes the trained model through a FastAPI inference API.

## Problem

Crop disease identification can be difficult to perform consistently from visual symptoms alone. This project explores an end-to-end computer-vision workflow from image preprocessing and transfer learning to API-based inference.

## System Flow

```text
Crop Image
    ↓
Image Preprocessing
    ↓
MobileNetV2 Transfer-Learning Model
    ↓
Disease / Damage Classification
    ↓
Confidence & Top Predictions
    ↓
Symptoms / Treatment Information
    ↓
FastAPI Response
```

## Highlights

- Multi-category crop disease and pest-damage classification
- Transfer learning with MobileNetV2
- Fine-tuning on approximately 14K training images
- Validation accuracy: **86.41%**
- Image upload inference API
- Confidence-aware predictions
- Top-2 prediction output
- Structured symptom, damage, and treatment information
- FastAPI backend with Swagger/OpenAPI documentation
- Data augmentation for more realistic image variation

## Model Categories

The current application works with categories including:

- Healthy
- Fungal Disease
- Bacterial Disease
- Viral Disease
- Pest Damage
- Rust Disease

## Datasets

Training used agricultural image datasets including PlantVillage and a multi-class crop-disease dataset.

The repository does not reproduce third-party dataset ownership. Dataset acquisition instructions should be followed according to the original dataset licenses and terms.

## Tech Stack

| Component | Technology |
|---|---|
| Language | Python |
| Deep learning | PyTorch |
| Computer vision | TorchVision, PIL |
| Model | MobileNetV2 |
| Backend | FastAPI |
| Server | Uvicorn |

## Project Structure

```text
pest-detection-system/
├── backend/
│   ├── api/
│   ├── services/
│   ├── knowledge-base/
│   └── main.py
├── ml-model/
│   ├── training/
│   └── saved_model/
├── PlantVillageDataset/
├── plant-village-dataset/
├── requirements.txt
├── .gitignore
└── README.md
```

## API

### Predict Disease

```http
POST /predict
```

The inference endpoint accepts a crop image and returns the predicted category, confidence, top predictions, and supporting symptom/treatment information.

Example response shape:

```json
{
  "filename": "leaf.jpg",
  "prediction": {
    "pest": "Viral_disease",
    "confidence": 92.41,
    "top_predictions": [
      {"class": "Viral_disease", "confidence": 92.41},
      {"class": "Rust_disease", "confidence": 5.82}
    ]
  }
}
```

## Engineering Focus

This project demonstrates:

- Transfer learning and fine-tuning
- Image augmentation
- Model inference packaging
- Confidence-aware prediction
- Serving a deep-learning model through a REST API
- Separation of model, backend, and knowledge-base components

## Future Improvements

- Grad-CAM explainability
- Better out-of-distribution evaluation
- Object detection for localized symptoms
- Mobile deployment
- Multilingual farmer support
- Real-world field-image evaluation

## Author

**Subham Dey**
