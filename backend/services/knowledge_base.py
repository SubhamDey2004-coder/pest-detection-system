import json
import os
# Get correct path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
KB_PATH = os.path.join(BASE_DIR, "knowledge-base", "pests.json")

def load_knowledge_base():
    with open(KB_PATH, "r") as f:
        return json.load(f)
    
kb_data = load_knowledge_base()

def get_pest_info(pest_name: str):
    pest_name = pest_name.lower()
    return kb_data.get(pest_name,{
        "symptoms": [],
        "damage": "No data available",
        "solution": []
    })
    
    