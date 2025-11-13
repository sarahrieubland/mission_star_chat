import chainlit as cl
from openai import AsyncOpenAI

# Initialize OpenAI client
client = AsyncOpenAI()

@cl.on_message
async def main(message: cl.Message):
    # Call OpenAI to generate a response
    response = await client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": message.content}
        ]
    )
    
    llm_output = response.choices[0].message.content
    
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
        content="Here's the generated response (editable on the right):",
        elements=[elem]
    ).send()