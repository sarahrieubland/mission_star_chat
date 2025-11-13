"""
POC: True Agentic STAR chatbot (Chainlit + LangChain Agents + Langfuse + OpenAI)

Updated to use LangChain's agent framework for true agentic behavior.
The agent can reason about what actions to take and use tools to accomplish its goals.

How to run (local POC):
1) Create virtual env and install requirements:
   pip install -r requirements_agent.txt

2) Set environment variables (example .env):
   OPENAI_API_KEY=sk-...
   LANGFUSE_API_KEY=lf_...
   LANGFUSE_PROJECT=my-project
   POC_API_KEY=secret-for-boss

3) Run with Chainlit (auto-reload):
   chainlit run star_agent_agentic.py -w

Description:
- Uses LangChain's ReAct agent pattern for true agentic reasoning
- Agent has access to tools: extract_star, check_completeness, ask_clarification
- Agent decides autonomously when to use each tool based on conversation state
- Implements agent memory for conversation context
- Integrates Langfuse for tracing agent decisions
"""

import os
import re
import json
import uuid
from typing import Dict, Any, Optional, List

import chainlit as cl
from dotenv import load_dotenv

# Modern LangChain imports
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# Import prompts from separate file
from config import prompts

# Show for debuggin
VERBOSE=True

# Langfuse callback handler
try:
    from langfuse.callback import CallbackHandler as LangfuseCallbackHandler
    LANGFUSE_AVAILABLE = True
except Exception:
    LANGFUSE_AVAILABLE = False

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
LANGFUSE_API_KEY = os.getenv("LANGFUSE_API_KEY")
LANGFUSE_PROJECT = os.getenv("LANGFUSE_PROJECT")

if not OPENAI_API_KEY:
    raise RuntimeError("Set OPENAI_API_KEY in env before running")

# --- In-memory session store (POC) ---
SESSIONS: Dict[str, Dict[str, Any]] = {}

# --- LLM setup ---
MODEL_NAME = os.getenv("POC_MODEL", "gpt-4")
TEMPERATURE = float(os.getenv("POC_TEMP", "0.0"))
VERBOSE = os.getenv("VERBOSE", "true").lower() == "true"  # Enable verbose output by default

# Attach Langfuse callback if available
lc_callbacks = []
if LANGFUSE_AVAILABLE and LANGFUSE_API_KEY and LANGFUSE_PROJECT:
    try:
        lf_cb = LangfuseCallbackHandler(
            public_key=LANGFUSE_API_KEY,
            secret_key=os.getenv("LANGFUSE_SECRET_KEY", ""),
            host=os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com")
        )
        lc_callbacks.append(lf_cb)
        print("Langfuse callback attached")
    except Exception as e:
        print("Could not initialize Langfuse callback:", e)

# --- Utility functions ---

def parse_json_from_text(text: str) -> Optional[Dict[str, Any]]:
    """Extract JSON object from LLM text response and parse it."""
    m = re.search(r"\{.*\}\s*$", text, re.DOTALL)
    if not m:
        m = re.search(r"\{.*?\}", text, re.DOTALL)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except Exception:
        s = m.group(0).replace("'", '"')
        s = re.sub(r",\s*\}", "}", s)
        try:
            return json.loads(s)
        except Exception:
            return None


def has_result_metric(text: str) -> bool:
    """Simple regex to detect numeric metrics in result text."""
    if not text:
        return False
    return bool(re.search(r"\d+\s*%|\d+\s*(users|people|days|months|years|k|m|M|\$|€|£)", text, re.IGNORECASE))


def identify_missing_fields(star: Dict[str, Any]) -> List[str]:
    """Return list of missing or weak fields."""
    missing = []
    priority = ["situation", "task", "action", "result"]
    for f in priority:
        entry = star.get(f, {})
        text = entry.get("text", "") if isinstance(entry, dict) else entry
        conf = entry.get("confidence", 0.0) if isinstance(entry, dict) else (0.0 if not text else 0.5)
        
        if not text or conf < 0.6:
            if f == "result" and text and has_result_metric(text):
                continue
            missing.append(f)
    return missing


def format_star_display(star: Dict[str, Any], verbose: bool = True) -> str:
    """Format STAR entry for display with optional verbose details."""
    if not star or "error" in star:
        return json.dumps(star, indent=2)
    
    if verbose:
        # Verbose format: show each section with its text and confidence
        output = "**📋 Entrée STAR actuelle:**\n\n"
        
        sections = {
            "situation": "**🎯 Situation**",
            "task": "**📌 Tâche**", 
            "action": "**⚡ Action**",
            "result": "**🎉 Résultat**"
        }
        
        for key, title in sections.items():
            entry = star.get(key, {})
            if isinstance(entry, dict):
                text = entry.get("text", "")
                conf = entry.get("confidence", 0.0)
                
                output += f"{title}\n"
                if text:
                    output += f"{text}\n"
                    output += f"_Confiance: {conf:.0%}_\n\n"
                else:
                    output += "_Non renseigné_\n\n"
            else:
                output += f"{title}\n{entry}\n\n"
        
        return output
    else:
        # Simple format: just JSON
        return f"```json\n{json.dumps(star, indent=2)}\n```"

# --- Agent Tools ---

@tool
def extract_star_from_description(description: str) -> str:
    """
    Extract STAR format (Situation, Task, Action, Result) from a job description.
    Returns a JSON string with the STAR structure including confidence scores.
    Use this tool when the user provides their initial job/task description.
    """
    llm = ChatOpenAI(model=MODEL_NAME, temperature=TEMPERATURE)
    
    prompt = prompts.EXTRACTION_PROMPT_TEMPLATE.format(input_text=description)
    response = llm.invoke([HumanMessage(content=prompt)])
    
    parsed = parse_json_from_text(response.content)
    if parsed:
        if VERBOSE:
            # Return formatted display
            return format_star_display(parsed, verbose=True)
        else:
            return json.dumps(parsed, indent=2)
    else:
        return json.dumps({"error": "Could not parse STAR format"})


@tool
def check_star_completeness(star_json: str) -> str:
    """
    Check if a STAR entry is complete and identify missing or weak fields.
    Returns a JSON object with 'is_complete' (boolean) and 'missing_fields' (list).
    Use this tool to evaluate if you need to ask follow-up questions.
    """
    try:
        star = json.loads(star_json)
        missing = identify_missing_fields(star)
        
        result = {
            "is_complete": len(missing) == 0,
            "missing_fields": missing,
            "analysis": f"Found {len(missing)} incomplete field(s)" if missing else "All fields are complete"
        }
        return json.dumps(result, indent=2)
    except Exception as e:
        return json.dumps({"error": str(e)})


@tool
def update_star_field_tool(star_json: str, field: str, new_information: str) -> str:
    """
    Update a specific field in the STAR entry with new information from the user.
    Args:
        star_json: The current STAR JSON string
        field: The field to update (situation, task, action, or result)
        new_information: The new information provided by the user
    Returns: Updated STAR JSON string
    """
    llm = ChatOpenAI(model=MODEL_NAME, temperature=TEMPERATURE)
    
    prompt = prompts.UPDATE_PROMPT_TEMPLATE.format(
        field=field,
        star_json=star_json,
        answer=new_information
    )
    
    response = llm.invoke([HumanMessage(content=prompt)])
    parsed = parse_json_from_text(response.content)
    
    if parsed:
        if VERBOSE:
            return format_star_display(parsed, verbose=True)
        else:
            return json.dumps(parsed, indent=2)
    else:
        return star_json  # Return original if update fails


@tool
def generate_clarifying_question(star_json: str, field: str, original_description: str) -> str:
    """
    Generate a targeted clarifying question for a specific missing or weak STAR field.
    Args:
        star_json: The current STAR JSON string
        field: The field that needs clarification
        original_description: The user's original job description
    Returns: A single clarifying question as a string
    """
    llm = ChatOpenAI(model=MODEL_NAME, temperature=TEMPERATURE)
    
    prompt = prompts.QUESTION_PROMPT_TEMPLATE.format(
        input_text=original_description,
        star_json=star_json,
        field=field
    )
    
    response = llm.invoke([HumanMessage(content=prompt)])
    return response.content.strip()

# --- Create Agentic LLM ---

def create_star_agent_llm(callbacks: List = None):
    """Create the STAR coaching LLM with tool calling capabilities."""
    
    llm = ChatOpenAI(model=MODEL_NAME, temperature=TEMPERATURE)
    
    # Define tools available to the agent
    tools = [
        extract_star_from_description,
        check_star_completeness,
        update_star_field_tool,
        generate_clarifying_question
    ]
    
    # Bind tools to the LLM (enables function calling)
    llm_with_tools = llm.bind_tools(tools)
    
    return llm_with_tools, tools


async def run_agent_loop(llm_with_tools, tools, user_input: str, chat_history: List, max_iterations: int = 10):
    """
    Run the agentic loop: reason -> act -> observe -> repeat
    This implements a simple ReAct pattern manually.
    """
    
    # Create tools dictionary for easy lookup
    tools_dict = {tool.name: tool for tool in tools}
    
    # Build the conversation with system prompt
    messages = [
        SystemMessage(content=prompts.AGENT_SYSTEM_PROMPT),
    ]
    messages.extend(chat_history)
    messages.append(HumanMessage(content=user_input))
    
    iteration = 0
    while iteration < max_iterations:
        iteration += 1
        
        if VERBOSE:
            print(f"\n🔄 Agent iteration {iteration}/{max_iterations}")
        
        # Agent reasoning: get response from LLM
        response = await llm_with_tools.ainvoke(messages)
        
        # Check if the LLM wants to use tools
        if not response.tool_calls:
            # No tool calls, agent is done - return final response
            if VERBOSE:
                print(f"✅ Agent completed in {iteration} iterations")
            return response.content
        
        # Execute tool calls
        messages.append(response)  # Add AI message with tool calls
        
        for tool_call in response.tool_calls:
            tool_name = tool_call["name"]
            tool_args = tool_call["args"]
            
            if VERBOSE:
                print(f"🔧 Calling tool: {tool_name}")
                print(f"   Args: {json.dumps(tool_args, indent=2)}")
            
            # Execute the tool
            if tool_name in tools_dict:
                try:
                    tool_result = await tools_dict[tool_name].ainvoke(tool_args)
                    
                    if VERBOSE:
                        print(f"✓ Tool result received (length: {len(str(tool_result))})")
                    
                    # Add tool result to messages
                    from langchain_core.messages import ToolMessage
                    messages.append(
                        ToolMessage(
                            content=str(tool_result),
                            tool_call_id=tool_call["id"]
                        )
                    )
                except Exception as e:
                    if VERBOSE:
                        print(f"❌ Tool error: {str(e)}")
                    messages.append(
                        ToolMessage(
                            content=f"Error executing tool: {str(e)}",
                            tool_call_id=tool_call["id"]
                        )
                    )
    
    # If we hit max iterations, return what we have
    if VERBOSE:
        print(f"⚠️ Reached max iterations ({max_iterations})")
    return "J'ai atteint ma limite d'itérations. Faites-moi savoir si vous souhaitez continuer!"

# --- Chainlit handlers ---

@cl.on_chat_start
async def on_chat_start():
    """Initialize agent and session."""
    session_id = str(uuid.uuid4())
    cl.user_session.set("session_id", session_id)
    
    # Create agent LLM with tools
    llm_with_tools, tools = create_star_agent_llm(callbacks=lc_callbacks)
    cl.user_session.set("llm_with_tools", llm_with_tools)
    cl.user_session.set("tools", tools)
    
    # Initialize conversation memory
    chat_history = []
    cl.user_session.set("chat_history", chat_history)
    
    # Initialize session data
    SESSIONS[session_id] = {
        "id": session_id,
        "star": None,
        "original_description": None
    }
    
    await cl.Message(content=prompts.WELCOME_MESSAGE).send()


@cl.on_message
async def main(message: cl.Message):
    """Handle incoming messages with the agent."""
    session_id = cl.user_session.get("session_id")
    if not session_id or session_id not in SESSIONS:
        await cl.Message(prompts.MSG_SESSION_NOT_FOUND).send()
        return
    
    # Get agent components and history
    llm_with_tools = cl.user_session.get("llm_with_tools")
    tools = cl.user_session.get("tools")
    chat_history = cl.user_session.get("chat_history", [])
    session = SESSIONS[session_id]
    
    # Store original description if this is the first message
    if not session.get("original_description"):
        session["original_description"] = message.content
    
    # Run the agent loop
    try:
        response_content = await run_agent_loop(
            llm_with_tools,
            tools,
            message.content,
            chat_history,
            max_iterations=10
        )
        
        # Update chat history
        chat_history.append(HumanMessage(content=message.content))
        chat_history.append(AIMessage(content=response_content))
        cl.user_session.set("chat_history", chat_history)
        
        # Send response
        await cl.Message(content=response_content).send()
        
    except Exception as e:
        await cl.Message(content=f"Une erreur s'est produite : {str(e)}").send()


if __name__ == "__main__":
    print("This file is intended to be run with: chainlit run star_agent_agentic.py -w")