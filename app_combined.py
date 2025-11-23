import os
import json
import chainlit as cl
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

# Import prompts from separate file
from config import prompts
from src.utils import star_json_to_txt

# --- LLM setup ---
MODEL_NAME = os.getenv("POC_MODEL", "gpt-4o-mini")
TEMPERATURE = float(os.getenv("POC_TEMP", "0.0"))
VERBOSE = os.getenv("VERBOSE", "true").lower() == "true"  # Enable verbose output by default

# Initialize LangChain ChatOpenAI
llm = ChatOpenAI(model=MODEL_NAME, temperature=TEMPERATURE)

@cl.on_chat_start
async def start():
    # Initialize session storage for saved STAR text
    cl.user_session.set("saved_star_text", None)
    cl.user_session.set("current_element", None)
    
    await cl.Message(
        content="""Bienvenu! Pour vous aider à generer la description de votre mission au format STAR, 
        veuillez tout d'abord décrire votre role et votre l'employeur (ou client) pour cette mission.
        Vous pouvez également insérer un texte déjà preparé."""
    ).send()

@cl.on_message
async def main(message: cl.Message):
    # Check if this is a save action from EditableText component
    if message.content.startswith("SAVE_STAR_TEXT:"):
        saved_text = message.content.replace("SAVE_STAR_TEXT:", "", 1)
        cl.user_session.set("saved_star_text", saved_text)
        
        # Show confirmation with the saved text
        if VERBOSE:
            await cl.Message(
                content=f"✓ Texte STAR sauvegardé!\n\n**Texte sauvegardé:**\n\n{saved_text}\n"
            ).send()
        else:
            await cl.Message(content="✓ Texte STAR sauvegardé!").send()
        
        return
    
    # Check if there's saved STAR text from previous interaction
    saved_star = cl.user_session.get("saved_star_text")
    #saved_star_json = star_txt_to_json(saved_star)
    
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
    
    # Send as a regular message so it stays visible
    await cl.Message(
        content="Voici un brouillon de votre mission au format STAR (à droite):",
        elements=[elem]
    ).send()


# Helper function to retrieve saved text (can be called from anywhere)
def get_saved_star_text():
    """Retrieve the saved STAR text from the session"""
    return cl.user_session.get("saved_star_text")