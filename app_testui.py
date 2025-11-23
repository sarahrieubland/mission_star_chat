import chainlit as cl
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

# Initialize LangChain ChatOpenAI
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)

@cl.on_chat_start
async def start():
    await cl.Message(
        content="""Bienvenu! Pour vous aider à generer la description de votre mission au format STAR, 
        veuillez tout d'abord décrire votre role et votre l'employeur (ou client) pour cette mission."""
    ).send()

@cl.on_message
async def main(message: cl.Message):
    # Use LangChain to generate a response
    messages = [
        SystemMessage(content="You are a helpful assistant."),
        HumanMessage(content=message.content)
    ]
    
    # Call the LLM
    response = await llm.ainvoke(messages)
    llm_output = response.content
    
    # Create editable text area using CustomElement
    elem = cl.CustomElement(
        name="EditableText",
        display="inline",
        props={
            "initial": llm_output,
            "keepVisible": True
        }
    )
    
    # Send as a regular message so it stays visible
    await cl.Message(
        content="Voici un brouillon de la description de votre mission au format STAR (à droite):",
        elements=[elem]
    ).send()