import os

import pandas as pd
from thefuzz import process


BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CSV_PATH = os.path.join(BASE_DIR, "data", "pesticide", "Pesticides.csv")

PEST_TYPE_MAPPING = {
    "Pest_damage": ["aphid", "army worm", "bollworm", "whitefly", "mite", "stem borer"],
    "Fungal_disease": ["rust", "blast", "smut", "wilt", "leaf spot"],
    "Bacterial_disease": ["bacterial blight"],
    "Viral_disease": ["mosaic", "leaf curl"],
    "Rust_disease": ["rust"],
    "Healthy": [],
}


def _load_pesticide_data():
    if not os.path.exists(CSV_PATH):
        return None
    return pd.read_csv(CSV_PATH)


df = _load_pesticide_data()


def search_pesticides(keyword, threshold=70):
    if df is None or "Pest Name" not in df.columns:
        return []

    pest_names = df["Pest Name"].dropna().astype(str).tolist()
    best_match = process.extractOne(keyword, pest_names, score_cutoff=threshold)

    if not best_match:
        return []

    matched_name = best_match[0]
    rows = df[df["Pest Name"] == matched_name]

    if "Most Commonly Used Pesticides" not in rows.columns:
        return []

    return (
        rows["Most Commonly Used Pesticides"]
        .dropna()
        .astype(str)
        .drop_duplicates()
        .tolist()
    )


def recommend_pesticides(predicted_class):
    recommendations = []

    for keyword in PEST_TYPE_MAPPING.get(predicted_class, []):
        recommendations.extend(search_pesticides(keyword))

    return list(dict.fromkeys(recommendations))[:5]
