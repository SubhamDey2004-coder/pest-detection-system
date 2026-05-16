import pandas as pd
from thefuzz import process
import os

# =========================
# LOAD CSV
# =========================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))

CSV_PATH = os.path.join(
    BASE_DIR,
    "data",
    "pesticide",
    "Pesticides.csv"
)

df = pd.read_csv(CSV_PATH)

# =========================
# CLASS → KEYWORD MAPPING
# =========================

PEST_TYPE_MAPPING = {

    "Pest_damage": [
        "aphid",
        "army worm",
        "bollworm",
        "whitefly",
        "mite",
        "stem borer"
    ],

    "Fungal_disease": [
        "rust",
        "blast",
        "smut",
        "wilt",
        "leaf spot"
    ],

    "Bacterial_disease": [
        "bacterial blight"
    ],

    "Viral_disease": [
        "mosaic",
        "leaf curl"
    ],

    "Rust_disease": [
        "rust"
    ],

    "Healthy": []
}


# =========================
# FUZZY SEARCH FUNCTION
# =========================

def search_pesticides(keyword, threshold=70):

    pest_names = df["Pest Name"].dropna().astype(str).tolist()

    best_match = process.extractOne(
        keyword,
        pest_names,
        score_cutoff=threshold
    )

    if not best_match:
        return []

    matched_name = best_match[0]

    rows = df[df["Pest Name"] == matched_name]

    recommendations = rows[
        "Most Commonly Used Pesticides"
    ].dropna().astype(str).tolist()

    # Remove duplicates
    recommendations = list(set(recommendations))

    return recommendations


# =========================
# MAIN RECOMMENDER
# =========================

def recommend_pesticides(predicted_class):

    recommendations = []

    keywords = PEST_TYPE_MAPPING.get(
        predicted_class,
        []
    )

    for keyword in keywords:

        results = search_pesticides(keyword)

        recommendations.extend(results)

    # Remove duplicates
    recommendations = list(set(recommendations))

    # Return top 5 only
    return recommendations[:5]