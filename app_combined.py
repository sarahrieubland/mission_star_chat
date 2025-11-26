import os
import json
import chainlit as cl
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from functools import wraps

# Import prompts from separate file
from config import prompts
from src.utils import star_json_to_txt, star_txt_to_json

# --- LLM setup ---
MODEL_NAME = os.getenv("POC_MODEL", "gpt-4o-mini")
TEMPERATURE = float(os.getenv("POC_TEMP", "0.0"))
VERBOSE = os.getenv("VERBOSE", "true").lower() == "true"  # Enable verbose output by default

# Initialize LangChain ChatOpenAI
llm = ChatOpenAI(model=MODEL_NAME, temperature=TEMPERATURE)

# Registry for custom event handlers
_event_handlers = {}

def on_star_text_saved(func):
    """
    Custom decorator to handle STAR text save events
    Usage:
        @on_star_text_saved
        async def my_handler(saved_text: str):
            # Your custom logic here
            pass
    """
    _event_handlers['star_text_saved'] = func
    @wraps(func)
    async def wrapper(saved_text: str):
        return await func(saved_text)
    return wrapper

@cl.on_chat_start
async def start():
    # Initialize session storage for saved STAR text
    cl.user_session.set("saved_star_text", None)
    cl.user_session.set("saved_star_json", None)
    cl.user_session.set("current_element", None)
    
    await cl.Message(
        content="""Bienvenu! Pour vous aider à generer la description de votre mission au format STAR, 
        veuillez tout d'abord décrire votre role et votre l'employeur (ou client) pour cette mission.
        Vous pouvez également insérer un texte déjà preparé."""
    ).send()

# Use the custom decorator - this is your "on_event" equivalent!
@on_star_text_saved
async def handle_save_action(saved_text: str):
    """Handle when user saves edited text - triggered by @on_star_text_saved decorator"""
    
    # Store the saved text
    cl.user_session.set("saved_star_text", saved_text)

    # Convert saved text into json
    saved_json = star_txt_to_json(saved_text)
    cl.user_session.set("saved_star_json", saved_json)
    
    # Update the EditableText element with the saved text
    elem = cl.user_session.get("current_element")
    if elem:
        elem.props["initial"] = saved_text
        await elem.update()
    
    # You can define any custom flow here after save
    
    if VERBOSE:
        # Safely get the Action field
        action_text = "N/A"
        if isinstance(saved_json, dict):
            if "action" in saved_json:
                action_text = saved_json["action"].get("text", "N/A") if isinstance(saved_json["action"], dict) else saved_json["action"]
            elif "Action" in saved_json:
                action_text = saved_json["Action"].get("text", "N/A") if isinstance(saved_json["Action"], dict) else saved_json["Action"]
        
        await cl.Message(
            content=f"✓ Texte STAR sauvegardé! Ex: Action - {action_text}"
        ).send()


@cl.on_message
async def main(message: cl.Message):
    # Check if this is a save event - route to registered handler
    if message.content.startswith("SAVE_STAR_TEXT:"):
        saved_text = message.content.replace("SAVE_STAR_TEXT:", "", 1)
        
        # Call the registered handler via decorator
        if 'star_text_saved' in _event_handlers:
            await _event_handlers['star_text_saved'](saved_text)
        return
    
    # Regular message flow
    await handle_user_message(message)


async def handle_user_message(message: cl.Message):
    """Handle regular user messages"""
    # Check if there's saved STAR text from previous interaction
    saved_star = cl.user_session.get("saved_star_text")
    
    if VERBOSE and saved_star:
        await cl.Message(content=f"📌 Utilisation du texte sauvegardé précédemment dans le contexte.").send()
    
    # Use LangChain to generate a response
    messages = [
        SystemMessage(content=prompts.EXTRACTION_PROMPT_TEMPLATE),
        HumanMessage(content=message.content)
    ]
    
    # If there's saved STAR text, include it in the context
    if saved_star:
        messages.insert(1, SystemMessage(content=f"Texte STAR précédemment sauvegardé par l'utilisateur:\n{saved_star}\n\nUtilisez ce contexte pour améliorer la réponse."))
    
    # Call the LLM
    response = await llm.ainvoke(messages)
    llm_output = response.content

    # Visualise the STAR format
    star_text = star_json_to_txt(llm_output)
    
    # Create editable text area using CustomElement
    elem = cl.CustomElement(
        name="EditableText",
        display="inline",
        props={
            "initial": star_text,
            "keepVisible": True
        }
    )
    
    # Store element reference so we can update it later
    cl.user_session.set("current_element", elem)
    
    # Send as a regular message so it stays visible
    await cl.Message(
        content="Voici un brouillon de votre mission au format STAR (à droite):",
        elements=[elem]
    ).send()


# Helper function to retrieve saved text (can be called from anywhere)
def get_saved_star_text():
    """Retrieve the saved STAR text from the session"""
    return cl.user_session.get("saved_star_text")


# Helper function to retrieve saved JSON (can be called from anywhere)
def get_saved_star_json():
    """Retrieve the saved STAR JSON from the session"""
    return cl.user_session.get("saved_star_json")