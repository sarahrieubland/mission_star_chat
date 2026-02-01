import re
import json


def star_json_to_txt(star_dict) -> str:
    """
    Converts a STAR dictionary or JSON string into formatted text.
    
    Expected format (dict or JSON string):
    {
        "situation": "text",
        "task": "text",
        "action": "text",
        "result": "text"
    }
    """
    # Handle both dict and string inputs
    if isinstance(star_dict, str):
        try:
            data = json.loads(star_dict)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON format: {e}")
    elif isinstance(star_dict, dict):
        data = star_dict
    else:
        raise TypeError("Input must be a dictionary or JSON string.")

    # Validate that data is a dictionary
    if not isinstance(data, dict):
        raise ValueError("Input must be a dictionary with STAR components.")

    # Build formatted output with PLURAL headers to match app
    text = (
        f"Situation:\n{data.get('situation', '').strip()}\n\n"
        f"Tâches:\n{data.get('task', '').strip()}\n\n"
        f"Actions:\n{data.get('action', '').strip()}\n\n"
        f"Résultats:\n{data.get('result', '').strip()}"
    )

    return text


def star_txt_to_json(text_str: str) -> dict:
    """Function to take the saved text from the user and put it back into json format"""

    # Updated regex to capture with PLURAL headers
    pattern = (
        r"Situation:\s*(.*?)\s*"
        r"Tâches:\s*(.*?)\s*"
        r"Actions:\s*(.*?)\s*"
        r"Résultats:\s*(.*)"
    )

    match = re.search(pattern, text_str, flags=re.DOTALL)
    if not match:
        return None

    situation, task, action, result = match.groups()

    return {
        "Situation": situation.strip(),
        "Tâches": task.strip(),
        "Actions": action.strip(),
        "Résultats": result.strip()
    }


def build_star_from_components(state):
    """Function to combine the components that are available in the state"""

    components = []
    if state.get('situation'):
        components.append(f"SITUATION: {state['situation']}")
    if state.get('task'):
        components.append(f"TÂCHE: {state['task']}")
    if state.get('action'):
        components.append(f"ACTION: {state['action']}")
    if state.get('result'):
        components.append(f"RÉSULTAT: {state['result']}")    
    components_text = "\n".join(components)

    return components, components_text