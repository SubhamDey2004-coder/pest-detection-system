# 🌱 AI-Powered Pest & Crop Disease Detection System

An AI-based agricultural disease analysis system that detects crop diseases and pest-related damage from uploaded crop images and provides treatment recommendations.

Built using **FastAPI**, **PyTorch**, and **Transfer Learning with MobileNetV2**.

---

# 🚀 Features

- 🌾 Multi-crop disease classification
- 🧠 Transfer learning using MobileNetV2
- 📸 Image upload API
- 📊 Confidence-aware predictions
- 🔍 Top-2 prediction analysis
- 🦠 Generalized disease categorization
- 💡 AI-generated treatment recommendations
- ⚡ FastAPI backend with Swagger documentation
- 🌍 Real-world image augmentation support

---

# 🧠 Disease Categories

The model classifies crop images into the following categories:

- Healthy
- Fungal Disease
- Bacterial Disease
- Viral Disease
- Pest Damage
- Rust Disease

---

# 🛠 Tech Stack

- Python
- FastAPI
- PyTorch
- TorchVision
- MobileNetV2
- PIL
- Uvicorn

---

# 📂 Project Structure

```plaintext
pest-detection-system/
│
├── backend/
│   ├── api/
│   ├── services/
│   ├── knowledge-base/
│   └── main.py
│
├── ml-model/
│   ├── training/
│   └── saved_model/
│
├── data/
│
├── requirements.txt
├── README.md
└── .gitignore
```

---

# 📊 Model Training

The model was trained using:

- PlantVillage Dataset
- 20k Multi-Class Crop Disease Dataset

Datasets:

- [PlantVillage Dataset](https://www.kaggle.com/datasets/emmarex/plantdisease?utm_source=chatgpt.com)
- [20k Multi-Class Crop Disease Dataset](https://www.kaggle.com/datasets/jawadali1045/20k-multi-class-crop-disease-images?utm_source=chatgpt.com)

---

# 🧪 Training Details

- Transfer Learning using MobileNetV2
- Fine-tuned on ~14k training images
- Validation Accuracy: **86.41%**
- Real-world augmentation:
  - Rotation
  - Blur
  - Color Jitter
  - Translation
  - Horizontal Flip

---

# ⚙ API Endpoint

## Predict Disease

```http
POST /predict
```

Upload a crop image and receive:
- predicted disease category
- confidence score
- top-2 predictions
- symptoms
- damage description
- treatment recommendations

---

# 📄 Example Response

```json
{
  "filename": "leaf.jpg",
  "prediction": {
    "pest": "Viral_disease",
    "confidence": 92.41,
    "warning": null,
    "top_predictions": [
      {
        "class": "Viral_disease",
        "confidence": 92.41
      },
      {
        "class": "Rust_disease",
        "confidence": 5.82
      }
    ],
    "symptoms": [
      "leaf curling",
      "mosaic patterns"
    ],
    "damage": "Viral diseases severely affect plant growth and productivity.",
    "solutions": [
      "Control insect vectors",
      "Remove infected plants",
      "Use resistant crop varieties"
    ]
  }
}
```

---

# 🧠 AI Engineering Highlights

- Unified disease abstraction instead of crop-specific memorization
- Confidence-aware AI predictions
- Transfer learning with fine-tuning
- Multi-dataset integration
- Robust real-world augmentation pipeline

---

# 🚀 Future Improvements

- Grad-CAM Explainable AI
- Object Detection (YOLO)
- Mobile App Integration
- Multilingual Farmer Support
- Real-Time Disease Detection

---

# 👨‍💻 Author

Subham Dey

---