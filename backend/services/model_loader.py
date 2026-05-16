import torch
import torch.nn as nn
from torchvision import models, transforms
import os


# IMPORTANT: same order as training (alphabetical from ImageFolder)
CLASS_NAMES = [
    'Bacterial_disease',
    'Fungal_disease',
    'Healthy',
    'Pest_damage',
    'Rust_disease',
    'Viral_disease'
]
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# Load model architecture
model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)

# Replace classifier
num_features = model.classifier[1].in_features
model.classifier[1] = nn.Linear(num_features, len(CLASS_NAMES))

# get absolute path to project root
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

MODEL_PATH = os.path.join(BASE_DIR, "ml-model", "saved_model", "pest_model_v3.pth")


# Load trained weights
model.load_state_dict(
    torch.load(MODEL_PATH, map_location=device)
)

model = model.to(device)
model.eval()

# Transform for image
transform = transforms.Compose([
    transforms.Resize((224, 244)),
    transforms.ToTensor()
])

def predict_image(image):
    # Transform image
    img_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(img_tensor)

    # Convert logits → probabilities
    probs = torch.softmax(outputs, dim=1)[0]

    # Get top 2 predictions
    top_probs, top_idxs = torch.topk(probs, 2)

    predictions = []

    for prob, idx in zip(top_probs, top_idxs):
        predictions.append({
            "class": CLASS_NAMES[idx.item()],
            "confidence": round(prob.item() * 100, 2)
        })

    return predictions