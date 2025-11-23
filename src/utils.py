
import json


def star_json_to_txt(json_str: str) -> str:
    """
    Converts a JSON string containing STAR elements into formatted text.
    
    Expected JSON format:
    {
        "situation": "text",
        "task": "text",
        "action": "text",
        "result": "text"
    }
    """
    # Validate input type
    if not isinstance(json_str, str):
        raise TypeError("Input must be a JSON string.")

    # Try to parse JSON
    try:
        data = json.loads(json_str)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON format: {e}")

    # Validate that data is a dictionary
    if not isinstance(data, dict):
        raise ValueError("Parsed JSON must be an object (dictionary).")

    # Expected STAR keys
    required_keys = {"situation", "task", "action", "result"}
    missing_keys = required_keys - data.keys()
    if missing_keys:
        raise ValueError(f"Missing required keys in JSON: {', '.join(missing_keys)}")

    # Build formatted output
    text = (
        f"Situation:\n{data.get('situation', '').strip()}\n\n"
        f"Tâche:\n{data.get('task', '').strip()}\n\n"
        f"Action:\n{data.get('action', '').strip()}\n\n"
        f"Résultat:\n{data.get('result', '').strip()}"
    )

    return text


def star_txt_to_json(text_str: str) -> dict:
    """Function to take the saved text from the user and put it back into json format"""

    json_star = dict()

    return json_star