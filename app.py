"""
STAR Text Generator - Agentic Workflow with LangGraph and Chainlit

This application helps users create and iteratively improve STAR 
(Situation, Task, Action, Result) descriptions for job interviews.

Features:
- LangGraph agentic workflow for gathering and improving STAR components
- Editable side panel showing the current STAR text at EVERY step
- User edits are incorporated back into the workflow
- Automatic save/submit from EditableText component
- Workflow continues after user saves edits (without regenerating text)
- Text format follows star_json_to_txt() format consistently
- LangSmith integration for tracing and monitoring
- LangSmith Hub for prompt management and versioning
"""

import asyncio
import json
import os
import uuid
from typing import TypedDict, Literal, Optional
from functools import wraps

# Load environment variables from .env file FIRST
from dotenv import load_dotenv
load_dotenv()

import chainlit as cl
from langsmith import Client
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, END

from src.utils import star_json_to_txt, star_txt_to_json, build_star_from_components
from src.prompt_manager import PromptManager


# --- Environment & Configuration ---
MODEL_NAME = os.getenv("POC_MODEL", "gpt-4o-mini")
TEMPERATURE = float(os.getenv("POC_TEMP", "0.0"))
VERBOSE = os.getenv("VERBOSE", "true").lower() == "true"

# LangSmith Hub configuration
# Set your LangSmith handle (username) here or in .env
LANGSMITH_HANDLE = os.getenv("LANGSMITH_HANDLE", "")  # e.g., "your-username"
USE_HUB_PROMPTS = os.getenv("USE_HUB_PROMPTS", "true").lower() == "true"

# --- LangSmith Configuration ---
os.environ.setdefault("LANGSMITH_TRACING", "true")

# Initialize LangSmith client
try:
    ls_client = Client()
    if VERBOSE:
        print("✅ LangSmith client initialized")
        print(f"   Project: {os.getenv('LANGSMITH_PROJECT', 'default')}")
        print(f"   Hub Handle: {LANGSMITH_HANDLE or 'Not set'}")
except Exception as e:
    ls_client = None
    if VERBOSE:
        print(f"⚠️ LangSmith client initialization failed: {e}")

# Generate a unique user ID for this session
user_id = f"user-{uuid.uuid4()}"

# Initialize LangChain OpenAI client
llm = ChatOpenAI(
    model=MODEL_NAME,
    temperature=TEMPERATURE,
)

# Registry for custom event handlers
_event_handlers = {}


# Initialize prompt manager
prompts = PromptManager(handle=LANGSMITH_HANDLE, use_hub=USE_HUB_PROMPTS, client=ls_client)


def extract_star_from_text(input_text: str) -> dict:
    """
    Extract STAR components from user's initial job description.
    
    This is a pure extraction step - no generation or embellishment.
    Returns a dict with keys: situation, task, action, result
    Empty strings for components not found in the input.
    """
    if VERBOSE:
        print("\n" + "="*50)
        print("🔍 INITIAL STAR EXTRACTION")
        print("="*50)
        print(f"   Input: {input_text[:200]}...")
    
    prompt = prompts.EXTRACTION_PROMPT.format(input_text=input_text)
    
    # Call LLM for extraction (no system prompt needed, instructions are in the prompt)
    response = call_llm(prompt, system="", run_name="initial_star_extraction")
    
    # Parse JSON response
    try:
        # Clean response if needed (remove markdown code blocks)
        cleaned = response.strip()
        if cleaned.startswith("```"):
            # Extract content between code blocks
            parts = cleaned.split("```")
            if len(parts) >= 2:
                cleaned = parts[1]
                if cleaned.startswith("json"):
                    cleaned = cleaned[4:]
        cleaned = cleaned.strip()
        
        extracted = json.loads(cleaned)
        
        # Ensure all keys exist
        result = {
            "situation": extracted.get("situation", "").strip(),
            "task": extracted.get("task", "").strip(),
            "action": extracted.get("action", "").strip(),
            "result": extracted.get("result", "").strip()
        }
        
        if VERBOSE:
            print("   ✅ Extraction successful:")
            print(f"      situation: {bool(result['situation'])} - '{result['situation'][:50]}...' " if result['situation'] else "      situation: (empty)")
            print(f"      task: {bool(result['task'])} - '{result['task'][:50]}...' " if result['task'] else "      task: (empty)")
            print(f"      action: {bool(result['action'])} - '{result['action'][:50]}...' " if result['action'] else "      action: (empty)")
            print(f"      result: {bool(result['result'])} - '{result['result'][:50]}...' " if result['result'] else "      result: (empty)")
            print("="*50 + "\n")
        
        return result
        
    except (json.JSONDecodeError, KeyError) as e:
        if VERBOSE:
            print(f"   ⚠️ Failed to parse extraction response: {e}")
            print(f"   Raw response: {response}")
            print("   📝 Returning empty components")
            print("="*50 + "\n")
        
        return {
            "situation": "",
            "task": "",
            "action": "",
            "result": ""
        }


class STARState(TypedDict):
    """State that persists across the workflow for the agent"""
    job_description: str
    situation: str
    task: str
    action: str
    result: str
    current_star_text: str
    pending_question: Optional[str]
    current_step: Literal["gather_info", "generate", "evaluate", "complete"]
    section_to_improve: Optional[Literal["situation", "task", "action", "result"]]
    iteration_count: int
    is_satisfactory: bool
    user_edited: bool
    skip_generate: bool


def on_star_text_saved(func):
    """Custom decorator to handle STAR text save events"""
    _event_handlers['star_text_saved'] = func
    @wraps(func)
    async def wrapper(saved_text: str):
        return await func(saved_text)
    return wrapper


def call_llm(prompt: str, system: str = None, run_name: str = None) -> str:
    """Make an LLM call with the given prompt."""
    if system is None:
        system = prompts.AGENT_SYSTEM_PROMPT
    
    messages = [
        SystemMessage(content=system),
        HumanMessage(content=prompt)
    ]
    
    if VERBOSE:
        print("\n" + "="*50)
        print(f"📤 LLM CALL{f' ({run_name})' if run_name else ''}")
        print("="*50)
        print(f"🔧 System: {system[:100]}..." if len(system) > 100 else f"🔧 System: {system}")
        print(f"📝 Prompt: {prompt[:200]}..." if len(prompt) > 200 else f"📝 Prompt: {prompt}")
        print("-"*50)
    
    response = llm.invoke(
        messages,
        config={
            "run_name": run_name or "llm_call",
            "metadata": {
                "user_id": user_id,
                "model": MODEL_NAME,
            }
        }
    )
    
    if VERBOSE:
        print(f"📥 Response: {response.content[:200]}..." if len(response.content) > 200 else f"📥 Response: {response.content}")
        print("="*50 + "\n")
    
    return response.content


def format_star_text_from_state(state: STARState) -> str:
    """Format the STAR text from state components."""
    situation = state.get('situation', '').strip()
    task = state.get('task', '').strip()
    action = state.get('action', '').strip()
    result = state.get('result', '').strip()
    
    text = (
        f"Situation:\n{situation}\n\n"
        f"Tâches:\n{task}\n\n"
        f"Actions:\n{action}\n\n"
        f"Résultats:\n{result}"
    )
    
    return text


def build_star_json_from_state(state: STARState) -> dict:
    """Build a JSON dict from state components."""
    return {
        "situation": state.get('situation', '').strip(),
        "task": state.get('task', '').strip(),
        "action": state.get('action', '').strip(),
        "result": state.get('result', '').strip()
    }


# Node functions for the LangGraph workflow

def gather_info_node(state: STARState) -> STARState:
    """Node that gathers information about a specific STAR component."""
    
    if VERBOSE:
        print("\n🔄 GATHER_INFO_NODE")
        print(f"   situation: '{state.get('situation', '')[:50]}' ({bool(state.get('situation'))})")
        print(f"   task: '{state.get('task', '')[:50]}' ({bool(state.get('task'))})")
        print(f"   action: '{state.get('action', '')[:50]}' ({bool(state.get('action'))})")
        print(f"   result: '{state.get('result', '')[:50]}' ({bool(state.get('result'))})")
        print(f"   section_to_improve: {state.get('section_to_improve')}")
        print(f"   user_edited: {state.get('user_edited', False)}")
    
    current_star_text = state.get("current_star_text", "")
    user_edited = state.get("user_edited", False)
    
    if not user_edited:
        has_any_component = any([
            state.get("situation"),
            state.get("task"),
            state.get("action"),
            state.get("result")
        ])
        
        if has_any_component:
            current_star_text = generate_star_text(state)
            if VERBOSE:
                print(f"   📝 Generated STAR text ({len(current_star_text)} chars)")
    else:
        if VERBOSE:
            print(f"   📝 Keeping user's saved text unchanged ({len(current_star_text)} chars)")
    
    section_to_improve = state.get("section_to_improve")
    components, components_text = build_star_from_components(state)
    
    if section_to_improve == "situation":
        prompt = prompts.SITUATION_PROMPT.format(input=components_text)
        question = call_llm(prompt, run_name="ask_improve_situation")
        if VERBOSE:
            print(f"   📌 Asking to improve SITUATION")
        return {**state, "pending_question": question, "section_to_improve": "situation", "current_star_text": current_star_text, "user_edited": False}
    
    elif section_to_improve == "task":
        prompt = prompts.TASK_PROMPT.format(input=components_text)
        question = call_llm(prompt, run_name="ask_improve_task")
        if VERBOSE:
            print(f"   📌 Asking to improve TASK")
        return {**state, "pending_question": question, "section_to_improve": "task", "current_star_text": current_star_text, "user_edited": False}
    
    elif section_to_improve == "action":
        prompt = prompts.ACTION_PROMPT.format(input=components_text)
        question = call_llm(prompt, run_name="ask_improve_action")
        if VERBOSE:
            print(f"   📌 Asking to improve ACTION")
        return {**state, "pending_question": question, "section_to_improve": "action", "current_star_text": current_star_text, "user_edited": False}
    
    elif section_to_improve == "result":
        prompt = prompts.RESULT_PROMPT.format(input=components_text)
        question = call_llm(prompt, run_name="ask_improve_result")
        if VERBOSE:
            print(f"   📌 Asking to improve RESULT")
        return {**state, "pending_question": question, "section_to_improve": "result", "current_star_text": current_star_text, "user_edited": False}
    
    # Normal flow: determine which component needs info
    if not state.get("situation"):
        prompt = prompts.SITUATION_PROMPT.format(input=state['job_description'])        
        question = call_llm(prompt, run_name="ask_situation")
        if VERBOSE:
            print(f"   📌 Asking for SITUATION (first time)")
        return {**state, "pending_question": question, "section_to_improve": "situation", "current_star_text": current_star_text, "user_edited": False}
    
    elif not state.get("task"):
        prompt = prompts.TASK_PROMPT.format(input=components_text)
        question = call_llm(prompt, run_name="ask_task")
        if VERBOSE:
            print(f"   📌 Asking for TASK (first time)")
        return {**state, "pending_question": question, "section_to_improve": "task", "current_star_text": current_star_text, "user_edited": False}
    
    elif not state.get("action"):
        prompt = prompts.ACTION_PROMPT.format(input=components_text)
        question = call_llm(prompt, run_name="ask_action")
        if VERBOSE:
            print(f"   📌 Asking for ACTION (first time)")
        return {**state, "pending_question": question, "section_to_improve": "action", "current_star_text": current_star_text, "user_edited": False}
    
    elif not state.get("result"):
        prompt = prompts.RESULT_PROMPT.format(input=components_text)
        question = call_llm(prompt, run_name="ask_result")
        if VERBOSE:
            print(f"   📌 Asking for RESULT (first time)")
        return {**state, "pending_question": question, "section_to_improve": "result", "current_star_text": current_star_text, "user_edited": False}
    
    if VERBOSE:
        print(f"   ✅ All components gathered")
    return {**state, "pending_question": None, "section_to_improve": None, "current_star_text": current_star_text, "user_edited": False}


def generate_star_text(state: STARState) -> str:
    """Generate STAR text from available components using LLM."""
    import re
    
    components, components_text = build_star_from_components(state)
    
    if not components:
        return format_star_text_from_state(state)
    
    if VERBOSE:
        print("   📝 GENERATING STAR TEXT FROM:")
        print(f"      Components: {len(components)}")
    
    prompt = prompts.GENERATE_STAR_PROMPT.format(input=components_text)
    llm_response = call_llm(prompt, run_name="generate_star_text")
    
    formatted_text = reformat_llm_response_to_standard(llm_response, state)
    
    return formatted_text


def reformat_llm_response_to_standard(llm_response: str, state: STARState) -> str:
    """Take LLM response and reformat it to the standard star_json_to_txt() format."""
    import re
    
    extracted = {
        "situation": "",
        "task": "",
        "action": "",
        "result": ""
    }
    
    parsed = star_txt_to_json(llm_response)
    if parsed:
        extracted["situation"] = parsed.get("Situation", parsed.get("situation", "")).strip()
        extracted["task"] = parsed.get("Tasks", parsed.get("s", parsed.get("task", ""))).strip()
        extracted["action"] = parsed.get("Actions", parsed.get("action", "")).strip()
        extracted["result"] = parsed.get("Results", parsed.get("Résultats", parsed.get("result", ""))).strip()
    else:
        patterns = {
            "situation": [r"situation\s*:", r"\*\*situation\*\*\s*:?", r"situation\s*\n"],
            "task": [r"tâche\s*:", r"task\s*:", r"\*\*tâche\*\*\s*:?", r"\*\*task\*\*\s*:?"],
            "action": [r"action\s*:", r"\*\*action\*\*\s*:?"],
            "result": [r"résultat\s*:", r"result\s*:", r"\*\*résultat\*\*\s*:?", r"\*\*result\*\*\s*:?"]
        }
        
        positions = []
        
        for key, pattern_list in patterns.items():
            for pattern in pattern_list:
                match = re.search(pattern, llm_response, re.IGNORECASE)
                if match:
                    positions.append((key, match.end()))
                    break
        
        positions.sort(key=lambda x: x[1])
        
        for i, (key, pos) in enumerate(positions):
            if i + 1 < len(positions):
                next_pos = positions[i + 1][1]
                for next_key, next_pattern_list in patterns.items():
                    if next_key == positions[i + 1][0]:
                        for pattern in next_pattern_list:
                            match = re.search(pattern, llm_response, re.IGNORECASE)
                            if match and match.end() == next_pos:
                                next_pos = match.start()
                                break
                        break
                content = llm_response[pos:next_pos].strip()
            else:
                content = llm_response[pos:].strip()
            
            content = re.sub(r'\*\*', '', content)
            content = content.strip()
            extracted[key] = content
    
    final = {
        "situation": extracted["situation"] if state.get("situation") else "",
        "task": extracted["task"] if state.get("task") else "",
        "action": extracted["action"] if state.get("action") else "",
        "result": extracted["result"] if state.get("result") else ""
    }
    
    if not any(final.values()) and any([state.get("situation"), state.get("task"), state.get("action"), state.get("result")]):
        final = {
            "situation": state.get("situation", "").strip(),
            "task": state.get("task", "").strip(),
            "action": state.get("action", "").strip(),
            "result": state.get("result", "").strip()
        }
    
    text = (
        f"Situation:\n{final['situation']}\n\n"
        f"Tâches:\n{final['task']}\n\n"
        f"Actions:\n{final['action']}\n\n"
        f"Résultats:\n{final['result']}"
    )
    
    return text


def generate_node(state: STARState) -> STARState:
    """Node that generates the final STAR text when all components are gathered."""
    
    if state.get("skip_generate", False):
        if VERBOSE:
            print("\n🔄 GENERATE_NODE - SKIPPED (user edited text)")
        return {
            **state, 
            "current_step": "evaluate",
            "skip_generate": False
        }
    
    if VERBOSE:
        print("\n🔄 GENERATE_NODE")
        print("-"*50)
        print("📋 CURRENT STATE:")
        print(f"   job_description: '{state.get('job_description', '')[:50]}...'")
        print(f"   situation: '{state.get('situation', '')[:50]}...'")
        print(f"   task: '{state.get('task', '')[:50]}...'")
        print(f"   action: '{state.get('action', '')[:50]}...'")
        print(f"   result: '{state.get('result', '')[:50]}...'")
        print("-"*50)
    
    star_text = generate_star_text(state)
    
    if VERBOSE:
        print("📥 GENERATED STAR TEXT:")
        print(star_text)
        print("="*50 + "\n")
    
    return {
        **state, 
        "current_star_text": star_text, 
        "current_step": "evaluate",
        "iteration_count": state.get("iteration_count", 0) + 1,
        "section_to_improve": None,
        "user_edited": False
    }


def evaluate_node(state: STARState) -> STARState:
    """Node that autonomously evaluates the STAR text."""
    
    if VERBOSE:
        print("\n🔄 EVALUATE_NODE")
        print(f"   iteration_count: {state.get('iteration_count', 0)}")
        print(f"   user_edited: {state.get('user_edited', False)}")
    
    all_components = all([
        state.get("situation"), 
        state.get("task"), 
        state.get("action"), 
        state.get("result")
    ])
    
    if VERBOSE:
        print(f"   all_components: {all_components}")
    
    if not all_components:
        if VERBOSE:
            print("   ⏭️ Not all components gathered, continuing to gather_info")
        return {**state, "current_step": "gather_info", "is_satisfactory": False, "section_to_improve": None}
    
    prompt = prompts.EVALUATE_PROMPT.format(input=state['current_star_text'])
    
    if VERBOSE:
        print("   📊 Evaluating STAR text...")
    
    response = call_llm(prompt, run_name="evaluate_star_text")
    
    try:
        cleaned = response.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.split("```")[1]
            if cleaned.startswith("json"):
                cleaned = cleaned[4:]
        cleaned = cleaned.strip()
        
        if VERBOSE:
            print(f"   📋 Cleaned evaluation response: {cleaned}")
        
        evaluation = json.loads(cleaned)
        is_satisfactory = evaluation.get("is_satisfactory", False)
        section_to_improve = evaluation.get("section_to_improve")
        question = evaluation.get("question")
        
        if VERBOSE:
            print(f"   ✅ is_satisfactory: {is_satisfactory}")
            print(f"   📌 section_to_improve: {section_to_improve}")
            print(f"   ❓ question: {question}")
        
        if is_satisfactory:
            return {
                **state, 
                "is_satisfactory": True, 
                "current_step": "complete", 
                "pending_question": None,
                "section_to_improve": None,
                "user_edited": False
            }
        else:
            return {
                **state, 
                "is_satisfactory": False, 
                "current_step": "gather_info",
                "pending_question": question,
                "section_to_improve": section_to_improve,
                "user_edited": False
            }
    
    except (json.JSONDecodeError, KeyError) as e:
        if VERBOSE:
            print(f"⚠️ Failed to parse evaluation response: {e}")
            print(f"   Raw response: {response}")
        return {
            **state, 
            "is_satisfactory": True, 
            "current_step": "complete", 
            "pending_question": None,
            "section_to_improve": None,
            "user_edited": False
        }


def complete_node(state: STARState) -> STARState:
    """Final node when the STAR text is complete"""
    if VERBOSE:
        print("\n✅ COMPLETE_NODE")
        print(f"   Final iteration count: {state.get('iteration_count', 0)}")
    return {**state, "current_step": "complete"}


# Router functions

def after_gather_router(state: STARState) -> str:
    """Router after gather_info"""
    all_components = all([
        state.get("situation"),
        state.get("task"),
        state.get("action"),
        state.get("result")
    ])
    
    if VERBOSE:
        print(f"\n🔀 AFTER_GATHER_ROUTER")
        print(f"   all_components: {all_components}")
        print(f"   pending_question: {bool(state.get('pending_question'))}")
        print(f"   skip_generate: {state.get('skip_generate', False)}")
    
    if state.get("pending_question"):
        if VERBOSE:
            print(f"   → END (waiting for user response)")
        return END
    elif all_components:
        if VERBOSE:
            print(f"   → generate")
        return "generate"
    else:
        if VERBOSE:
            print(f"   → END (waiting for more components)")
        return END


def after_evaluate_router(state: STARState) -> str:
    """Router after evaluate"""
    
    if VERBOSE:
        print(f"\n🔀 AFTER_EVALUATE_ROUTER")
        print(f"   is_satisfactory: {state.get('is_satisfactory')}")
        print(f"   section_to_improve: {state.get('section_to_improve')}")
        print(f"   pending_question: {bool(state.get('pending_question'))}")
    
    if state.get("is_satisfactory"):
        if VERBOSE:
            print(f"   → complete")
        return "complete"
    elif state.get("pending_question"):
        if VERBOSE:
            print(f"   → END (waiting for user to answer improvement question)")
        return END
    else:
        if VERBOSE:
            print(f"   → gather_info")
        return "gather_info"


# Build the LangGraph workflow
def create_star_graph():
    """Create and compile the LangGraph workflow"""
    
    workflow = StateGraph(STARState)
    
    workflow.add_node("gather_info", gather_info_node)
    workflow.add_node("generate", generate_node)
    workflow.add_node("evaluate", evaluate_node)
    workflow.add_node("complete", complete_node)
    
    workflow.set_entry_point("gather_info")
    
    workflow.add_conditional_edges(
        "gather_info",
        after_gather_router,
        {
            "generate": "generate",
            END: END
        }
    )
    
    workflow.add_edge("generate", "evaluate")
    
    workflow.add_conditional_edges(
        "evaluate",
        after_evaluate_router,
        {
            "complete": "complete",
            "gather_info": "gather_info",
            END: END
        }
    )
    
    workflow.add_edge("complete", END)
    
    return workflow.compile()


# Create the graph instance
graph = create_star_graph()


# --- EditableText Helper Functions ---

def parse_star_text_to_components(star_text: str) -> dict:
    """Parse STAR text back into individual components."""
    components = {
        "situation": "",
        "task": "",
        "action": "",
        "result": ""
    }
    
    parsed = star_txt_to_json(star_text)
    
    if parsed:
        components["situation"] = parsed.get("Situation", "").strip()
        components["task"] = parsed.get("Tasks", "").strip()
        components["action"] = parsed.get("Action", "").strip()
        components["result"] = parsed.get("Results", "").strip()
    
    if VERBOSE:
        print(f"   Parsed components from text:")
        print(f"      situation: {bool(components['situation'])}")
        print(f"      task: {bool(components['task'])}")
        print(f"      action: {bool(components['action'])}")
        print(f"      result: {bool(components['result'])}")
    
    return components


async def update_editable_text(star_text: str):
    """Update the EditableText element with new STAR text"""
    elem = cl.user_session.get("current_element")
    
    if elem:
        elem.props["initial"] = star_text
        await elem.update()
        if VERBOSE:
            print("📝 Updated EditableText element")
    else:
        elem = cl.CustomElement(
            name="EditableText",
            display="inline",
            props={"initial": star_text, "keepVisible": True}
        )
        cl.user_session.set("current_element", elem)
        if VERBOSE:
            print("📝 Created new EditableText element")
    
    return elem


# --- Chainlit Event Handlers ---

@on_star_text_saved
async def handle_save_action(saved_text: str):
    """Handle when user saves edited text from the side panel."""
    if VERBOSE:
        print("\n" + "="*50)
        print("💾 STAR TEXT SAVED BY USER")
        print("="*50)
        print(f"   Saved text length: {len(saved_text)}")
    
    cl.user_session.set("saved_star_text", saved_text)
    
    parsed = star_txt_to_json(saved_text)
    cl.user_session.set("saved_star_json", parsed)
    
    components = parse_star_text_to_components(saved_text)
    
    if VERBOSE:
        print("   📋 Parsed components:")
        print(f"      situation: {bool(components.get('situation'))} - '{components.get('situation', '')[:30]}...'")
        print(f"      task: {bool(components.get('task'))} - '{components.get('task', '')[:30]}...'")
        print(f"      action: {bool(components.get('action'))} - '{components.get('action', '')[:30]}...'")
        print(f"      result: {bool(components.get('result'))} - '{components.get('result', '')[:30]}...'")
    
    state = cl.user_session.get("state")
    if state:
        state["situation"] = components.get("situation", "")
        state["task"] = components.get("task", "")
        state["action"] = components.get("action", "")
        state["result"] = components.get("result", "")
        
        state["current_star_text"] = saved_text
        state["user_edited"] = True
        state["skip_generate"] = True
        state["pending_question"] = None
        state["section_to_improve"] = None
        state["current_step"] = "gather_info"
        
        if VERBOSE:
            print("   ✅ State updated with user edits")
            print("   🔒 User's text will be preserved (skip_generate=True)")
        
        cl.user_session.set("state", state)
    
    await cl.Message(content="✅ Modifications sauvegardées! Évaluation en cours...").send()
    
    if VERBOSE:
        print("\n⚙️ Re-invoking graph after user edit (will skip generation)...")
    
    result = await asyncio.to_thread(
        graph.invoke, 
        state,
        config={
            "run_name": "star_workflow_after_user_edit",
            "metadata": {
                "user_id": cl.user_session.get("user_id", user_id),
                "trigger": "user_save"
            }
        }
    )
    
    if VERBOSE:
        print(f"✅ Graph invocation complete after user edit.")
        print(f"   New step: {result.get('current_step')}")
        print(f"   section_to_improve: {result.get('section_to_improve')}")
        print(f"   is_satisfactory: {result.get('is_satisfactory')}")
        print(f"   current_star_text preserved: {result.get('current_star_text') == saved_text}")
    
    cl.user_session.set("state", result)
    
    elem = cl.user_session.get("current_element")
    
    if result.get("current_step") == "complete":
        final_message = f"""🎉 **Votre mission STAR est terminée !**

Votre texte a été validé.

🌟 Vous pouvez encore modifier le texte dans le panneau de droite si nécessaire."""
        
        if elem:
            await cl.Message(content=final_message, elements=[elem]).send()
        else:
            await cl.Message(content=final_message).send()
    
    elif result.get("pending_question"):
        section_name = result.get("section_to_improve", "").upper() if result.get("section_to_improve") else ""
        
        if section_name:
            question_msg = f"""📝 Merci pour vos modifications!

🔍 **Pour améliorer la section {section_name}:**

{result.get('pending_question')}"""
        else:
            question_msg = f"""📝 Merci pour vos modifications!

{result.get('pending_question')}"""
        
        if elem:
            await cl.Message(content=question_msg, elements=[elem]).send()
        else:
            await cl.Message(content=question_msg).send()
    
    else:
        await cl.Message(content="📝 Modifications prises en compte. Continuez à répondre aux questions pour améliorer votre texte STAR.").send()


# --- Password-based Authentication ---
@cl.password_auth_callback
def auth_callback(username: str, password: str) -> cl.User | None:
    """Verify username and password for login.
    
    Set APP_USERNAME and APP_PASSWORD in .env file.
    If not set, authentication is skipped.
    """
    expected_username = os.getenv("APP_USERNAME", "")
    expected_password = os.getenv("APP_PASSWORD", "")
    
    # If no credentials configured, allow all access
    if not expected_username or not expected_password:
        return cl.User(identifier="anonymous", metadata={"role": "user"})
    
    # Check credentials
    if username == expected_username and password == expected_password:
        return cl.User(identifier=username, metadata={"role": "user"})
    
    # Authentication failed
    return None


@cl.on_chat_start
async def start():
    """Initialize the chat session"""
    
    session_id = f"session-{uuid.uuid4()}"
    
    initial_state: STARState = {
        "job_description": "",
        "situation": "",
        "task": "",
        "action": "",
        "result": "",
        "current_star_text": "",
        "pending_question": None,
        "current_step": "gather_info",
        "section_to_improve": None,
        "iteration_count": 0,
        "is_satisfactory": False,
        "user_edited": False,
        "skip_generate": False
    }

    cl.user_session.set("user_id", user_id)
    cl.user_session.set("session_id", session_id)
    cl.user_session.set("state", initial_state)
    cl.user_session.set("saved_star_text", None)
    cl.user_session.set("saved_star_json", None)
    cl.user_session.set("current_element", None)
    cl.user_session.set("extraction_done", False)  # Track if initial extraction is done
    
    if VERBOSE:
        print(f"\n🚀 New session started")
        print(f"   user_id: {user_id}")
        print(f"   session_id: {session_id}")
    
    welcome = prompts.WELCOME_MESSAGE
    await cl.Message(content=welcome).send()


@cl.on_message
async def main(message: cl.Message):
    """Handle incoming messages."""
    
    if message.content.startswith("SAVE_STAR_TEXT:"):
        saved_text = message.content.replace("SAVE_STAR_TEXT:", "", 1)
        if 'star_text_saved' in _event_handlers:
            await _event_handlers['star_text_saved'](saved_text)
        return
    
    await handle_workflow_message(message)


async def handle_workflow_message(message: cl.Message):
    """Handle regular user messages through the LangGraph workflow.
    
    Flow:
    1. First message: Extract STAR components from job description
    2. Subsequent messages: Process through LangGraph workflow
    """
    
    state = cl.user_session.get("state")
    extraction_done = cl.user_session.get("extraction_done", False)
    
    current_step = state.get("current_step", "gather_info")
    section_to_improve = state.get("section_to_improve")
    
    if VERBOSE:
        print("\n" + "="*50)
        print("📨 USER MESSAGE RECEIVED")
        print("="*50)
        print(f"   extraction_done: {extraction_done}")
        print(f"   current_step: {current_step}")
        print(f"   section_to_improve: {section_to_improve}")
        print(f"   user_input: {message.content[:100]}...")
    
    # --- FIRST MESSAGE: Initial Extraction ---
    if not extraction_done:
        if VERBOSE:
            print("\n🔍 First message - performing initial STAR extraction...")
        
        # Store the job description
        state["job_description"] = message.content
        
        # Show processing message
        processing_msg = await cl.Message(content="🔍 Analyse de votre description en cours...").send()
        
        # Extract STAR components from the initial input
        extracted = await asyncio.to_thread(extract_star_from_text, message.content)
        
        # Store extraction result in session
        cl.user_session.set("initial_extraction", extracted)
        
        # Update state with extracted components
        state["situation"] = extracted.get("situation", "")
        state["task"] = extracted.get("task", "")
        state["action"] = extracted.get("action", "")
        state["result"] = extracted.get("result", "")
        
        # Generate initial STAR text from extracted components
        if any([state["situation"], state["task"], state["action"], state["result"]]):
            state["current_star_text"] = format_star_text_from_state(state)
        
        # Mark extraction as done
        cl.user_session.set("extraction_done", True)
        cl.user_session.set("state", state)
        
        if VERBOSE:
            print(f"   ✅ Initial extraction complete")
            print(f"      situation: {bool(state['situation'])}")
            print(f"      task: {bool(state['task'])}")
            print(f"      action: {bool(state['action'])}")
            print(f"      result: {bool(state['result'])}")
        
        # Show the extracted STAR text in the editable panel
        elem = None
        if state.get("current_star_text"):
            elem = await update_editable_text(state["current_star_text"])
        
        # Build summary message for the user
        extracted_sections = []
        if state["situation"]:
            extracted_sections.append("Situation ✓")
        if state["task"]:
            extracted_sections.append("Tâches ✓")
        if state["action"]:
            extracted_sections.append("Actions ✓")
        if state["result"]:
            extracted_sections.append("Résultats ✓")
        
        missing_sections = []
        if not state["situation"]:
            missing_sections.append("Situation")
        if not state["task"]:
            missing_sections.append("Tâches")
        if not state["action"]:
            missing_sections.append("Actions")
        if not state["result"]:
            missing_sections.append("Résultats")
        
        extraction_summary = f"""✅ **Extraction initiale terminée !**

**Éléments identifiés :** {', '.join(extracted_sections) if extracted_sections else 'Aucun'}
"""
        if missing_sections:
            extraction_summary += f"\n**Éléments à compléter :** {', '.join(missing_sections)}"
        
        extraction_summary += "\n\n📝 Le texte STAR est affiché dans le panneau de droite. Je vais maintenant vous poser quelques questions pour l'améliorer."
        
        if elem:
            await cl.Message(content=extraction_summary, elements=[elem]).send()
        else:
            await cl.Message(content=extraction_summary).send()
        
        # Now invoke the LangGraph workflow to start the improvement process
        if VERBOSE:
            print("\n⚙️ Invoking LangGraph workflow after extraction...")
        
        result = await asyncio.to_thread(
            graph.invoke, 
            state,
            config={
                "run_name": "star_workflow_after_extraction",
                "metadata": {
                    "user_id": cl.user_session.get("user_id", user_id),
                    "session_id": cl.user_session.get("session_id", "unknown"),
                    "trigger": "initial_extraction",
                }
            }
        )
        
        if VERBOSE:
            print(f"✅ Graph invocation complete after extraction.")
            print(f"   New step: {result.get('current_step')}")
            print(f"   section_to_improve: {result.get('section_to_improve')}")
            print(f"   pending_question: {bool(result.get('pending_question'))}")
        
        # Update session state
        cl.user_session.set("state", result)
        
        # Update EditableText if text changed
        if result.get("current_star_text") and result.get("current_star_text") != state.get("current_star_text"):
            elem = await update_editable_text(result["current_star_text"])
        
        # Send the first question from the workflow
        if result.get("pending_question"):
            section_name = result.get("section_to_improve", "").upper() if result.get("section_to_improve") else ""
            
            if section_name:
                question_msg = f"""🔍 **Section {section_name}:**

{result.get('pending_question')}"""
            else:
                question_msg = result.get('pending_question')
            
            if elem:
                await cl.Message(content=question_msg, elements=[elem]).send()
            else:
                await cl.Message(content=question_msg).send()
        
        return
    
    # --- SUBSEQUENT MESSAGES: LangGraph Workflow ---
    saved_star = cl.user_session.get("saved_star_text")
    if VERBOSE and saved_star:
        print(f"   📌 Previously saved text available in context")
    
    if section_to_improve:
        if section_to_improve == "situation":
            state["situation"] = state.get("situation", "") + " " + message.content
            if VERBOSE:
                print(f"   → Appended to SITUATION")
        elif section_to_improve == "task":
            state["task"] = state.get("task", "") + " " + message.content
            if VERBOSE:
                print(f"   → Appended to TASK")
        elif section_to_improve == "action":
            state["action"] = state.get("action", "") + " " + message.content
            if VERBOSE:
                print(f"   → Appended to ACTION")
        elif section_to_improve == "result":
            state["result"] = state.get("result", "") + " " + message.content
            if VERBOSE:
                print(f"   → Appended to RESULT")
        state["section_to_improve"] = None
        state["pending_question"] = None
        state["user_edited"] = False
        state["skip_generate"] = False
        
    elif current_step == "gather_info":
        if not state.get("situation"):
            state["situation"] = message.content
            if VERBOSE:
                print(f"   → Stored as SITUATION")
        elif not state.get("task"):
            state["task"] = message.content
            if VERBOSE:
                print(f"   → Stored as TASK")
        elif not state.get("action"):
            state["action"] = message.content
            if VERBOSE:
                print(f"   → Stored as ACTION")
        elif not state.get("result"):
            state["result"] = message.content
            if VERBOSE:
                print(f"   → Stored as RESULT")
    
    if VERBOSE:
        print("\n⚙️ Invoking graph...")
    
    result = await asyncio.to_thread(
        graph.invoke, 
        state,
        config={
            "run_name": "star_workflow",
            "metadata": {
                "user_id": cl.user_session.get("user_id", user_id),
                "session_id": cl.user_session.get("session_id", "unknown"),
                "current_step": current_step,
            }
        }
    )
    
    if VERBOSE:
        print(f"✅ Graph invocation complete.")
        print(f"   New step: {result.get('current_step')}")
        print(f"   section_to_improve: {result.get('section_to_improve')}")
        print(f"   is_satisfactory: {result.get('is_satisfactory')}")
        print(f"   current_star_text length: {len(result.get('current_star_text', ''))}")
    
    cl.user_session.set("state", result)
    
    elem = None
    if result.get("current_star_text"):
        elem = await update_editable_text(result["current_star_text"])
        if VERBOSE:
            print(f"   📝 EditableText updated with {len(result['current_star_text'])} chars")
    
    if result.get("current_step") == "complete":
        final_message = f"""🎉 **Votre mission STAR est terminée !**

Voici votre mission STAR finale (après {result.get('iteration_count', 1)} itération(s)) :

---

{result.get('current_star_text', '')}

---

🌟 Vous pouvez encore modifier le texte dans le panneau de droite si nécessaire."""
        
        if elem:
            await cl.Message(content=final_message, elements=[elem]).send()
        else:
            await cl.Message(content=final_message).send()
    
    elif result.get("pending_question"):
        section_name = result.get("section_to_improve", "").upper() if result.get("section_to_improve") else ""
        
        if section_name:
            question_msg = f"""🔍 **Section {section_name}:**

{result.get('pending_question')}"""
        else:
            question_msg = result.get('pending_question')
        
        if elem:
            await cl.Message(content=question_msg, elements=[elem]).send()
        else:
            await cl.Message(content=question_msg).send()


# --- Utility Functions for external access ---

def get_saved_star_text():
    """Get the saved STAR text from the session"""
    return cl.user_session.get("saved_star_text")


def get_saved_star_json():
    """Get the saved STAR JSON from the session"""
    return cl.user_session.get("saved_star_json")


def get_current_state() -> STARState:
    """Get the current workflow state"""
    return cl.user_session.get("state")


if __name__ == "__main__":
    # This file is run by chainlit, not directly
    # To push prompts to hub, run:
    # python -c "from app import push_all_prompts_to_hub; push_all_prompts_to_hub()"
    pass