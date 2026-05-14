import os
import shutil

# =========================
# SOURCE DATASET PATHS
# =========================

TRAIN_SOURCE = "../../data/crop-disease-20k/Train"
VAL_SOURCE = "../../data/crop-disease-20k/Validation"

# =========================
# TARGET PATHS
# =========================

TRAIN_TARGET = "../../data/cleaned_v2/train"
VAL_TARGET = "../../data/cleaned_v2/val"

# =========================
# CLASS MAPPING
# =========================

CLASS_MAPPING = {

    # -------------------------
    # HEALTHY
    # -------------------------
    "Healthy": [
        "Healthy cotton",
        "Healthy Maize",
        "Sugarcane Healthy"
    ],

    # -------------------------
    # FUNGAL DISEASES
    # -------------------------
    "Fungal_disease": [
        "Anthracnose on Cotton",
        "Brownspot",
        "Flag Smut",
        "Gray_Leaf_Spot",
        "Leaf smut",
        "Rice Blast",
        "Wheat black rust",
        "Wheat Brown leaf Rust",
        "Wheat leaf blight",
        "Wheat powdery mildew",
        "Wheat scab",
        "Wilt"
    ],

    # -------------------------
    # BACTERIAL DISEASES
    # -------------------------
    "Bacterial_disease": [
        "Becterial Blight in Rice",
        "bacterial blight in Cotton"
    ],

    # -------------------------
    # PEST DAMAGE
    # -------------------------
    "Pest_damage": [
        "American Bollworm on Cotton",
        "Army worm",
        "bollworm on Cotton",
        "Cotton Aphid",
        "cotton mealy bug",
        "cotton whitefly",
        "maize fall armyworm",
        "maize stem borer",
        "pink bollworm in cotton",
        "red cotton bug",
        "thirps on cotton",
        "Wheat aphid",
        "Wheat mite",
        "Wheat Stem fly"
    ],

    # -------------------------
    # VIRAL DISEASES
    # -------------------------
    "Viral_disease": [
        "Leaf Curl",
        "Mosaic sugarcane",
        "Tungro"
    ],

    # -------------------------
    # RUST DISEASES
    # -------------------------
    "Rust_disease": [
        "Common_Rust",
        "RedRust sugarcane",
        "Wheat___Yellow_Rust",
        "Yellow Rust Sugarcane"
    ]
}

# =========================
# FUNCTION TO PROCESS DATA
# =========================

def process_dataset(source_dir, target_dir):
    total_copied = 0
    
    for new_class, old_classes in CLASS_MAPPING.items():
        new_class_path = os.path.join(target_dir, new_class)
        os.makedirs(new_class_path, exist_ok=True)
        
        print(f"\n📁 Processing: {new_class}")
        
        for old_class in old_classes:
            
            old_path = os.path.join(source_dir, old_class)
            
            if not os.path.exists(old_path):
                print(f"❌ Missing folder: {old_path}")
                continue
            
            files = os.listdir(old_path)
            
            print(f"✅ Found: {old_class} → {len(files)} images")
            
            for img in files:
                src = os.path.join(old_path, img)
                
                # Prevent duplicate overwriting
                dst = os.path.join(new_class_path, f"{old_class}_{img}")
                
                try:
                    shutil.copy(src, dst)
                    total_copied += 1
                except Exception as e:
                    print(f"⚠️ Error copying {img}: {e}")
    print("\n✅ Dataset processing complete!")
    print(f"📊 Total copied: {total_copied}")
    
# =========================
# RUN TRAIN + VAL
# =========================

print("\n================ TRAIN DATA ================\n")
process_dataset(TRAIN_SOURCE, TRAIN_TARGET)

print("\n================ VALIDATION DATA ================\n")
process_dataset(VAL_SOURCE, VAL_TARGET)