from services.model_loader import predict_image
from services.knowledge_base import get_pest_info
from services.pesticide_recommender import recommend_pesticides


def predict(image):

    # =========================
    # GET ML PREDICTIONS
    # =========================

    predictions = predict_image(image)

    top_prediction = predictions[0]

    pest_name = top_prediction["class"]
    confidence = top_prediction["confidence"]

    # =========================
    # LOW CONFIDENCE WARNING
    # =========================

    warning = None

    if confidence < 75:
        warning = (
            "Prediction confidence is low. "
            "Please upload a clearer image."
        )

    # =========================
    # GET KNOWLEDGE BASE INFO
    # =========================

    pest_info = get_pest_info(pest_name.lower())

    # =========================
    # GET PESTICIDE RECOMMENDATIONS
    # =========================

    pesticides = recommend_pesticides(pest_name)

    # =========================
    # FINAL RESPONSE
    # =========================

    return {

        "pest": pest_name,

        "confidence": confidence,

        "warning": warning,

        "top_predictions": predictions,

        "symptoms": pest_info.get(
            "symptoms",
            []
        ),

        "damage": pest_info.get(
            "damage",
            "No data available"
        ),

        "solutions": pest_info.get(
            "solutions",
            []
        ),

        "recommended_pesticides": pesticides
    }