import os

import torch
import torch.nn as nn
from torchvision import models, transforms


CLASS_NAMES = [
    "Bacterial_disease",
    "Fungal_disease",
    "Healthy",
    "Pest_damage",
    "Rust_disease",
    "Viral_disease",
]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
num_features = model.classifier[1].in_features
model.classifier[1] = nn.Linear(num_features, len(CLASS_NAMES))

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODEL_PATH = os.path.join(BASE_DIR, "ml-model", "saved_model", "pest_model_v3.pth")

model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
model = model.to(device)
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])


def predict_image(image):
    img_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(img_tensor)

    probabilities = torch.softmax(outputs, dim=1)[0]
    top_probs, top_indices = torch.topk(probabilities, k=2)

    return [
        {
            "class": CLASS_NAMES[index.item()],
            "confidence": round(probability.item() * 100, 2),
        }
        for probability, index in zip(top_probs, top_indices)
    ]
