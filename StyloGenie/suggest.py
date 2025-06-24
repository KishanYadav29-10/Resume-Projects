import json
from utils import format_list

def suggest_outfit(clothing_item: str, color: str, occasion: str = "", user_prefs: dict = None) -> str:
    """
    Suggest outfit combinations based on color and context, with memory-based personalization.

    Args:
        clothing_item (str): e.g., "shirt", "kurta"
        color (str): main color mentioned, e.g., "green"
        occasion (str): e.g., "party", "interview" (optional)
        user_prefs (dict): optional memory object with previous clothing info

    Returns:
        str: Outfit suggestion
    """
    try:
        with open("color_rules.json", "r") as f:
            color_data = json.load(f)
    except FileNotFoundError:
        return "Color rules data is missing."

    color = color.lower()
    suggestion = []

    # 🧠 Personalized message based on last worn color
    if user_prefs:
        last_color = user_prefs.get("last_color", "")
        if last_color and last_color != color:
            suggestion.append(f"Last time, you wore something {last_color}. Let's switch it up!")

    suggestion.append(f"You're wearing a {color} {clothing_item}.")

    if color not in color_data:
        suggestion.append("I don't have outfit suggestions for that color yet.")
        return " ".join(suggestion)

    rules = color_data[color]
    complementary = rules.get("complementary", "")
    neutral = rules.get("neutral_matches", [])
    analogs = rules.get("analogous", [])

    if complementary and complementary != "none":
        suggestion.append(f"Try pairing it with something {complementary} for contrast.")
    
    if neutral:
        suggestion.append(f"Neutral tones like {format_list(neutral)} also go well with {color}.")
    
    if analogs:
        suggestion.append(f"For a more harmonious look, you can try {format_list(analogs)} tones.")

    if occasion:
        suggestion.append(f"Since it's for a {occasion.lower()}, consider keeping it {occasion_tip(occasion)}.")

    return " ".join(suggestion)

def occasion_tip(occasion: str) -> str:
    """
    Return style tips based on occasion.
    """
    tips = {
        "party": "vibrant and expressive",
        "interview": "subtle and professional",
        "date": "stylish yet comfortable",
        "college": "trendy and relaxed",
        "wedding": "traditional and festive"
    }
    return tips.get(occasion.lower(), "appropriate and comfortable")
