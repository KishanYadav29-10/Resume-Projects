# utils.py

def format_list(items):
    """
    Format a list into a readable comma-separated string with 'and'.
    Example: ['red', 'blue', 'green'] => 'red, blue, and green'
    """
    if not items:
        return ""
    if len(items) == 1:
        return items[0]
    return ", ".join(items[:-1]) + f", and {items[-1]}"

def capitalize_first(text: str) -> str:
    """
    Capitalize the first character of a sentence.
    """
    return text[0].upper() + text[1:] if text else ""

def is_color_known(color: str, color_data: dict) -> bool:
    """
    Check if the given color exists in the color_rules.json dictionary.
    """
    return color.lower() in color_data

def clean_input(text: str) -> str:
    """
    Clean user input (strip whitespace, fix casing).
    """
    return text.strip().lower()
