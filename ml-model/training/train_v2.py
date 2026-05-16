import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
import os

# =========================
# DEVICE
# =========================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =========================
# DATA PATHS
# =========================

DATA_DIR = "../../data/cleaned_v2"

TRAIN_DIR = os.path.join(DATA_DIR, "train")
VAL_DIR = os.path.join(DATA_DIR, "val")

# =========================
# TRANSFORMS
# =========================

train_transform = transforms.Compose([

    transforms.Resize((224, 224)),

    # Flip images randomly
    transforms.RandomHorizontalFlip(),

    # Random rotation
    transforms.RandomRotation(20),

    # -------------------------
    # BLACK & WHITE AUGMENTATION
    # -------------------------
    # 25% images become grayscale
    transforms.RandomGrayscale(p=0.25),

    # -------------------------
    # NIGHT / LOW LIGHT SIMULATION
    # -------------------------
    transforms.ColorJitter(
        brightness=0.15,
        contrast=0.3,
        saturation=0.3
    ),

    # Slight image movement
    transforms.RandomAffine(
        degrees=0,
        translate=(0.1, 0.1),
        scale=(0.9, 1.1)
    ),

    # Slight blur
    transforms.GaussianBlur(kernel_size=3),

    # Simulate camera blur/noise
    transforms.RandomAdjustSharpness(
        sharpness_factor=0.5,
        p=0.3
    ),

    transforms.ToTensor()
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# =========================
# DATASETS
# =========================

train_data = datasets.ImageFolder(TRAIN_DIR, transform=train_transform)
val_data = datasets.ImageFolder(VAL_DIR, transform=val_transform)

print("\n📊 Class Order:")
print(train_data.classes)

train_loader = torch.utils.data.DataLoader(
    train_data,
    batch_size=32,
    shuffle=True
)

val_loader = torch.utils.data.DataLoader(
    val_data,
    batch_size=32
)

# =========================
# MODEL
# =========================

model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)

# Replace classifier
num_features = model.classifier[1].in_features

model.classifier[1] = nn.Linear(
    num_features,
    len(train_data.classes)
)

# =========================
# LOAD PREVIOUS MODEL
# =========================

PREVIOUS_MODEL_PATH = "../saved_model/pest_model.pth"

if os.path.exists(PREVIOUS_MODEL_PATH):

    print("\n🔁 Loading previous model weights...")

    try:
        old_state = torch.load(
            PREVIOUS_MODEL_PATH,
            map_location=device
        )

        model_state = model.state_dict()

        # Load matching layers only
        filtered_state = {
            k: v
            for k, v in old_state.items()
            if k in model_state and v.size() == model_state[k].size()
        }

        model_state.update(filtered_state)

        model.load_state_dict(model_state)

        print("✅ Previous weights loaded successfully!")

    except Exception as e:
        print(f"⚠️ Could not load previous weights: {e}")

else:
    print("⚠️ No previous model found. Training from scratch.")

model = model.to(device)

# =========================
# LOSS + OPTIMIZER
# =========================

criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(
    model.parameters(),
    lr=0.0002
)

# =========================
# TRAINING LOOP
# =========================

EPOCHS = 5

best_accuracy = 0

for epoch in range(EPOCHS):

    # -------------------------
    # TRAIN
    # -------------------------

    model.train()

    running_loss = 0.0

    for images, labels in train_loader:

        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(outputs, labels)

        loss.backward()

        optimizer.step()

        running_loss += loss.item() * images.size(0)

    epoch_loss = running_loss / len(train_loader.dataset)

    # -------------------------
    # VALIDATION
    # -------------------------

    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():

        for images, labels in val_loader:

            images, labels = images.to(device), labels.to(device)

            outputs = model(images)

            _, predicted = torch.max(outputs, 1)

            total += labels.size(0)

            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total

    print(
        f"\nEpoch [{epoch+1}/{EPOCHS}] "
        f"Loss: {epoch_loss:.4f} "
        f"| Val Accuracy: {accuracy:.2f}%"
    )

    # -------------------------
    # SAVE BEST MODEL
    # -------------------------

    if accuracy > best_accuracy:

        best_accuracy = accuracy

        os.makedirs("../saved_model", exist_ok=True)

        torch.save(
            model.state_dict(),
            "../saved_model/pest_model_v3.pth"
        )

        print("💾 Best model updated!")

# =========================
# FINAL MESSAGE
# =========================

print("\n✅ Advanced model training complete!")
print(f"🏆 Best Validation Accuracy: {best_accuracy:.2f}%")