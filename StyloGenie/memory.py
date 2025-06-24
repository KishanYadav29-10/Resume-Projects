import json
import os

# File to store memory data
MEMORY_FILE = "user_memory.json"

def load_memory() -> dict:
    """
    Load memory from the JSON file.
    Returns a dictionary containing all user memory.
    """
    if os.path.exists(MEMORY_FILE):
        with open(MEMORY_FILE, "r") as file:
            return json.load(file)
    return {}

def save_memory(memory: dict):
    """
    Save the updated memory dictionary to the JSON file.
    """
    with open(MEMORY_FILE, "w") as file:
        json.dump(memory, file, indent=4)

def update_memory(user_id: str, entry: dict):
    """
    Update memory for a specific user with new key-value pairs.

    Args:
        user_id (str): Unique identifier for the user.
        entry (dict): Dictionary of preferences or data to update.
    """
    memory = load_memory()
    user_data = memory.get(user_id, {})
    user_data.update(entry)
    memory[user_id] = user_data
    save_memory(memory)

def get_user_preferences(user_id: str) -> dict:
    """
    Get stored preferences for a user.

    Args:
        user_id (str): Unique identifier for the user.
    Returns:
        dict: User's stored preferences or an empty dict.
    """
    memory = load_memory()
    return memory.get(user_id, {})

def clear_memory(user_id: str):
    """
    Clear stored memory for a specific user.

    Args:
        user_id (str): Unique identifier for the user.
    """
    memory = load_memory()
    if user_id in memory:
        del memory[user_id]
        save_memory(memory)
