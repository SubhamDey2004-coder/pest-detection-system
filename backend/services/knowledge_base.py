import json
import os


BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
KB_PATH = os.path.join(BASE_DIR, "knowledge-base", "pests.json")


def load_knowledge_base():
    with open(KB_PATH, "r", encoding="utf-8") as file:
        return json.load(file)


kb_data = load_knowledge_base()


def get_pest_info(pest_name: str):
    return kb_data.get(
        pest_name.lower(),
        {
            "symptoms": [],
            "damage": "No data available",
            "solutions": [],
        },
    )
