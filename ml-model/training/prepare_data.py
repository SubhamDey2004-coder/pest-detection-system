import os
import shutil

# 🔁 Change this for train / val
SOURCE_DIR = "../../data/val"
TARGET_DIR = "../../data/cleaned/val"

CLASS_MAPPING = {
    "Early_blight": [
        "Potato___Early_blight"
    ],
    "Late_blight": [
        "Potato___Late_blight"
    ],
    "Bacterial_spot": [
        "Pepper__bell___Bacterial_spot"
    ],
    "Healthy": [
        "Potato___healthy",
        "Pepper__bell___healthy"
    ]
}

total_copied = 0

for new_class, old_classes in CLASS_MAPPING.items():
    new_class_path = os.path.join(TARGET_DIR, new_class)
    os.makedirs(new_class_path, exist_ok=True)

    print(f"\n📁 Processing class: {new_class}")

    for old_class in old_classes:
        old_path = os.path.join(SOURCE_DIR, old_class)

        if not os.path.exists(old_path):
            print(f"❌ Missing folder: {old_path}")
            continue

        files = os.listdir(old_path)
        print(f"✅ Found: {old_class} → {len(files)} images")

        for img in files:
            src = os.path.join(old_path, img)

            # Prevent overwrite by adding prefix
            dst = os.path.join(new_class_path, f"{old_class}_{img}")

            try:
                shutil.copy(src, dst)
                total_copied += 1
            except Exception as e:
                print(f"⚠️ Error copying {img}: {e}")

print("\n✅ Dataset preparation complete!")
print(f"📊 Total images copied: {total_copied}")