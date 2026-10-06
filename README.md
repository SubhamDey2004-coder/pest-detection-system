# AI-Powered Pest & Crop Disease Detection System

An image-based agricultural computer-vision system that classifies crop disease and pest-related damage and serves predictions through a FastAPI API.

## What it does

1. Accepts a crop image through an API endpoint.
2. Preprocesses the image for MobileNetV2.
3. Predicts one of six disease / damage categories.
4. Returns the top-2 predictions with confidence scores.
5. Adds symptom, damage, and solution information from a local JSON knowledge base.
6. Optionally returns pesticide recommendations when the local pesticide dataset is available.

## Model

The project uses **MobileNetV2 transfer learning** with six output classes:

- Bacterial_disease
- Fungal_disease
- Healthy
- Pest_damage
- Rust_disease
- Viral_disease

The API loads the trained weights from:

~~~
ml-model/saved_model/pest_model_v3.pth
~~~

Training uses image augmentation, including horizontal flips, rotations, grayscale conversion, color jitter, affine transforms, blur, and sharpness adjustment.

## System Flow

~~~text
Crop Image
    ↓
Image Preprocessing
    ↓
MobileNetV2
    ↓
Disease / Damage Classification
    ↓
Top-2 Predictions + Confidence
    ↓
Knowledge Base + Optional Pesticide Recommendations
    ↓
FastAPI JSON Response
~~~

## API

### Health / root

GET /

### Test route

GET /test

### Predict

POST /predict

The prediction endpoint accepts an image upload using the **file** form field.

Example with curl:

~~~bash
curl -X POST "http://127.0.0.1:8000/predict" -F "file=@leaf.jpg"
~~~

FastAPI also exposes interactive API documentation at /docs.

## Setup

### 1. Create an environment

~~~bash
python -m venv .venv
~~~

Windows:

~~~powershell
.\.venv\Scripts\Activate.ps1
~~~

### 2. Install dependencies

~~~bash
pip install -r requirements.txt
~~~

### 3. Start the API

From the repository root:

~~~bash
cd backend
uvicorn main:app --reload
~~~

The API will be available at http://127.0.0.1:8000.

## Project Structure

~~~text
pest-detection-system/
├── backend/
│   ├── api/
│   │   └── routes.py
│   ├── services/
│   │   ├── knowledge_base.py
│   │   ├── model_loader.py
│   │   ├── pesticide_recommender.py
│   │   └── predictor.py
│   ├── utils/
│   │   └── image_processing.py
│   └── main.py
├── knowledge-base/
│   └── pests.json
├── ml-model/
│   ├── saved_model/
│   │   └── pest_model_v3.pth
│   └── training/
│       ├── prepare_data_v2.py
│       ├── train.py
│       └── train_v2.py
├── requirements.txt
├── .gitignore
└── README.md
~~~

## Training

The active training pipeline is in ml-model/training/train_v2.py. It expects prepared training and validation data under:

~~~text
data/
└── cleaned_v2/
    ├── train/
    └── val/
~~~

The dataset itself is not included in this repository. prepare_data_v2.py can reorganize the referenced crop-disease dataset into the expected class structure.

The repository includes the trained pest_model_v3.pth artifact so the API can be run without retraining.

## Engineering Highlights

- PyTorch transfer learning with MobileNetV2
- Image augmentation for training
- FastAPI model-serving API
- Top-k confidence scoring
- Local JSON knowledge base
- Optional fuzzy pesticide lookup using pandas and TheFuzz
- CPU/GPU device selection for inference
- Separation of model, API, service, and training components

## Limitations

- The model is intended as a project prototype, not a replacement for agronomist diagnosis.
- Real-world field images can differ substantially from curated training data.
- Pesticide recommendations depend on the optional local pesticide dataset and should be validated against crop, pest, dosage, regulations, and local agricultural guidance.
- Dataset files are intentionally not committed to the repository.

## Future Improvements

- Grad-CAM explainability
- Out-of-distribution and field-image evaluation
- Better calibration of confidence scores
- Object detection for localized symptoms
- Mobile deployment
- Multilingual farmer support

## Author

**Subham Dey**
