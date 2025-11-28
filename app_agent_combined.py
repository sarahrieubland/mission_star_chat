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
"""

import asyncio
import json
import os
import uuid
from typing import TypedDict, Literal, Optional
from functools import wraps

import chainlit as cl
from langsmith import Client
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, END

from config import prompts
from src.utils import star_json_to_txt, star_txt_to_json, build_star_from_components

from dotenv import load_dotenv
load_dotenv()

# --- Environment & Configuration ---
MODEL_NAME = os.getenv("POC_MODEL", "gpt-4o-mini")
TEMPERATURE = float(os.getenv("POC_TEMP", "0.0"))
VERBOSE = os.getenv("VERBOSE", "true").lower() == "true"

# --- LangSmith Configuration ---
os.environ.setdefault("LANGSMITH_TRACING", "true")

# Initialize LangSmith client for additional operations (optional)
try:
    ls_client = Client()
    if VERBOSE:
        print("✅ LangSmith client initialized")
        print(f"   Project: {os.getenv('LANGSMITH_PROJECT', 'default')}")
except Exception as e:
    ls_client = None
    if VERBOSE:
        print(f"⚠️ LangSmith client initialization failed: {e}")

# Generate a unique user ID for this session
user_id = f"user-{uuid.uuid4()}"

# Initialize LangChain OpenAI client
# LangSmith will automatically trace all calls made through this client
llm = ChatOpenAI(
    model=MODEL_NAME,
    temperature=TEMPERATURE,
)

# Registry for custom event handlers
_event_handlers = {}


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
    user_edited: bool  # Flag to indicate user made edits - skip regeneration
    skip_generate: bool  # Flag to skip generate node and go straight to evaluate


def on_star_text_saved(func):
    """Custom decorator to handle STAR text save events"""
    _event_handlers['star_text_saved'] = func
    @wraps(func)
    async def wrapper(saved_text: str):
        return await func(saved_text)
    return wrapper


def call_llm(prompt: str, system: str = prompts.AGENT_SYSTEM_PROMPT, run_name: str = None) -> str:
    """Make an LLM call with the given prompt.
    
    LangSmith automatically traces this call when LANGSMITH_TRACING=true.
    The run_name parameter allows you to give a descriptive name to the trace.
    """
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
    
    # LangSmith will automatically trace this invoke call
    # You can add metadata for better organization in the LangSmith UI
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
    """
    Format the STAR text from state components using the standard format.
    This ensures consistent formatting matching star_json_to_txt().
    
    Empty sections are shown as empty (not invented).
    """
    situation = state.get('situation', '').strip()
    task = state.get('task', '').strip()
    action = state.get('action', '').strip()
    result = state.get('result', '').strip()
    
    text = (
        f"Situation:\n{situation}\n\n"
        f"Tâche:\n{task}\n\n"
        f"Action:\n{action}\n\n"
        f"Résultat:\n{result}"
    )
    
    return text


def build_star_json_from_state(state: STARState) -> dict:
    """
    Build a JSON dict from state components.
    Empty sections are included as empty strings.
    """
    return {
        "situation": state.get('situation', '').strip(),
        "task": state.get('task', '').strip(),
        "action": state.get('action', '').strip(),
        "result": state.get('result', '').strip()
    }


# Node functions for the LangGraph workflow

def gather_info_node(state: STARState) -> STARState:
    """Node that gathers information about a specific STAR component.
    
    This node generates STAR text after asking for info, so the
    EditableText panel is updated at every step.
    
    IMPORTANT: 
    - If user_edited is True, we do NOT regenerate - we keep the user's saved text
    - Text format follows star_json_to_txt() format
    - Empty sections remain empty (LLM does not invent content)
    """
    
    if VERBOSE:
        print("\n🔄 GATHER_INFO_NODE")
        print(f"   situation: '{state.get('situation', '')[:50]}' ({bool(state.get('situation'))})")
        print(f"   task: '{state.get('task', '')[:50]}' ({bool(state.get('task'))})")
        print(f"   action: '{state.get('action', '')[:50]}' ({bool(state.get('action'))})")
        print(f"   result: '{state.get('result', '')[:50]}' ({bool(state.get('result'))})")
        print(f"   section_to_improve: {state.get('section_to_improve')}")
        print(f"   user_edited: {state.get('user_edited', False)}")
    
    # Get current star text - DO NOT regenerate if user edited
    current_star_text = state.get("current_star_text", "")
    user_edited = state.get("user_edited", False)
    
    # Only generate if user hasn't edited and we have components
    if not user_edited:
        has_any_component = any([
            state.get("situation"),
            state.get("task"),
            state.get("action"),
            state.get("result")
        ])
        
        if has_any_component:
            # Generate STAR text using LLM but format output consistently
            current_star_text = generate_star_text(state)
            if VERBOSE:
                print(f"   📝 Generated STAR text ({len(current_star_text)} chars)")
    else:
        if VERBOSE:
            print(f"   📝 Keeping user's saved text unchanged ({len(current_star_text)} chars)")
    
    # If evaluate told us which section to improve, ask about that specific section
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
    
    # Normal flow: determine which component needs info (first time through)
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
    
    # All components gathered, clear section_to_improve and move to generate
    if VERBOSE:
        print(f"   ✅ All components gathered")
    return {**state, "pending_question": None, "section_to_improve": None, "current_star_text": current_star_text, "user_edited": False}


def generate_star_text(state: STARState) -> str:
    """
    Generate STAR text from available components using LLM.
    
    IMPORTANT: 
    - The LLM should ONLY reformulate what's provided, NOT invent content
    - The output is then formatted to match star_json_to_txt() format
    - Empty sections remain empty
    """
    components, components_text = build_star_from_components(state)
    
    if not components:
        # No components yet - return empty template
        return format_star_text_from_state(state)
    
    if VERBOSE:
        print("   📝 GENERATING STAR TEXT FROM:")
        print(f"      Components: {len(components)}")
    
    # Call LLM to reformulate (not invent) the content
    prompt = prompts.GENERATE_STAR_PROMPT.format(input=components_text)
    llm_response = call_llm(prompt, run_name="generate_star_text")
    
    # Try to parse LLM response and reformat to standard format
    # The LLM might return in various formats, so we try to extract and reformat
    formatted_text = reformat_llm_response_to_standard(llm_response, state)
    
    return formatted_text


def reformat_llm_response_to_standard(llm_response: str, state: STARState) -> str:
    """
    Take LLM response and reformat it to the standard star_json_to_txt() format.
    
    This ensures:
    - Consistent format: "Situation:\n...\n\nTâche:\n...\n\nAction:\n...\n\nRésultat:\n..."
    - Empty sections remain empty (not filled with invented content)
    - No markdown formatting like **SITUATION**
    """
    import re
    
    # Try to parse the LLM response to extract sections
    extracted = {
        "situation": "",
        "task": "",
        "action": "",
        "result": ""
    }
    
    # Try using star_txt_to_json first
    parsed = star_txt_to_json(llm_response)
    if parsed:
        # Map the parsed keys to our standard keys
        extracted["situation"] = parsed.get("Situation", parsed.get("situation", "")).strip()
        extracted["task"] = parsed.get("Tasks", parsed.get("Tâche", parsed.get("task", ""))).strip()
        extracted["action"] = parsed.get("Action", parsed.get("action", "")).strip()
        extracted["result"] = parsed.get("Results", parsed.get("Résultat", parsed.get("result", ""))).strip()
    else:
        # Fallback: try to extract sections manually
        # Find positions of each section (handle various formats)
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
        
        # Sort by position
        positions.sort(key=lambda x: x[1])
        
        # Extract content between positions
        for i, (key, pos) in enumerate(positions):
            if i + 1 < len(positions):
                next_pos = positions[i + 1][1]
                # Find the start of the next section header
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
            
            # Clean up the content (remove markdown, extra newlines, etc.)
            content = re.sub(r'\*\*', '', content)  # Remove **
            content = content.strip()
            extracted[key] = content
    
    # IMPORTANT: Only include content that was actually in the original state
    # This prevents LLM from inventing content for empty sections
    final = {
        "situation": extracted["situation"] if state.get("situation") else "",
        "task": extracted["task"] if state.get("task") else "",
        "action": extracted["action"] if state.get("action") else "",
        "result": extracted["result"] if state.get("result") else ""
    }
    
    # If extraction failed but we have state content, use state directly
    if not any(final.values()) and any([state.get("situation"), state.get("task"), state.get("action"), state.get("result")]):
        final = {
            "situation": state.get("situation", "").strip(),
            "task": state.get("task", "").strip(),
            "action": state.get("action", "").strip(),
            "result": state.get("result", "").strip()
        }
    
    # Format to standard output
    text = (
        f"Situation:\n{final['situation']}\n\n"
        f"Tâche:\n{final['task']}\n\n"
        f"Action:\n{final['action']}\n\n"
        f"Résultat:\n{final['result']}"
    )
    
    return text


def generate_node(state: STARState) -> STARState:
    """Node that generates the final STAR text when all components are gathered.
    
    IMPORTANT: If skip_generate is True (after user edit), we skip regeneration
    and keep the user's saved text.
    """
    
    # Check if we should skip generation (user edited the text)
    if state.get("skip_generate", False):
        if VERBOSE:
            print("\n🔄 GENERATE_NODE - SKIPPED (user edited text)")
        return {
            **state, 
            "current_step": "evaluate",
            "skip_generate": False  # Reset flag
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
    
    # Generate STAR text
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
    """Node that autonomously evaluates the STAR text and decides which section needs improvement.
    
    This node does NOT modify the STAR text - it only evaluates and decides next steps.
    """
    
    if VERBOSE:
        print("\n🔄 EVALUATE_NODE")
        print(f"   iteration_count: {state.get('iteration_count', 0)}")
        print(f"   user_edited: {state.get('user_edited', False)}")
    
    # Check if all components are gathered
    all_components = all([
        state.get("situation"), 
        state.get("task"), 
        state.get("action"), 
        state.get("result")
    ])
    
    if VERBOSE:
        print(f"   all_components: {all_components}")
    
    if not all_components:
        # Still gathering info, continue to next component
        if VERBOSE:
            print("   ⏭️ Not all components gathered, continuing to gather_info")
        return {**state, "current_step": "gather_info", "is_satisfactory": False, "section_to_improve": None}
    
    # All components gathered - evaluate the text (without modifying it)
    prompt = prompts.EVALUATE_PROMPT.format(input=state['current_star_text'])
    
    if VERBOSE:
        print("   📊 Evaluating STAR text...")
    
    response = call_llm(prompt, run_name="evaluate_star_text")
    
    # Parse the response
    try:
        # Clean response if needed
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
        # Default to satisfactory if parsing fails
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
    """Router after gather_info - check if all components gathered"""
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
        # We have a question to ask, stop and wait for user
        if VERBOSE:
            print(f"   → END (waiting for user response)")
        return END
    elif all_components:
        # All components gathered
        if VERBOSE:
            print(f"   → generate")
        return "generate"
    else:
        if VERBOSE:
            print(f"   → END (waiting for more components)")
        return END


def after_evaluate_router(state: STARState) -> str:
    """Router after evaluate to decide next step"""
    
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
        # Evaluate provided a question, stop and wait for user
        if VERBOSE:
            print(f"   → END (waiting for user to answer improvement question)")
        return END
    else:
        # Continue gathering info
        if VERBOSE:
            print(f"   → gather_info")
        return "gather_info"


# Build the LangGraph workflow
def create_star_graph():
    """Create and compile the LangGraph workflow"""
    
    # Create the graph
    workflow = StateGraph(STARState)
    
    # Add nodes
    workflow.add_node("gather_info", gather_info_node)
    workflow.add_node("generate", generate_node)
    workflow.add_node("evaluate", evaluate_node)
    workflow.add_node("complete", complete_node)
    
    # Set entry point
    workflow.set_entry_point("gather_info")
    
    # After gathering info: either wait for user or go to generate if all done
    workflow.add_conditional_edges(
        "gather_info",
        after_gather_router,
        {
            "generate": "generate",
            END: END
        }
    )
    
    # After generating: always evaluate
    workflow.add_edge("generate", "evaluate")
    
    # After evaluating: complete, wait for user, or continue gathering
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
    
    # Compile WITHOUT checkpointer - we manage state via Chainlit session
    return workflow.compile()


# Create the graph instance
graph = create_star_graph()


# --- EditableText Helper Functions ---

def parse_star_text_to_components(star_text: str) -> dict:
    """
    Parse STAR text back into individual components using star_txt_to_json.
    Returns dict with keys: situation, task, action, result
    """
    components = {
        "situation": "",
        "task": "",
        "action": "",
        "result": ""
    }
    
    # Use the utility function
    parsed = star_txt_to_json(star_text)
    
    if parsed:
        # Map the parsed keys to our standard lowercase keys
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
        # Update existing element
        elem.props["initial"] = star_text
        await elem.update()
        if VERBOSE:
            print("📝 Updated EditableText element")
    else:
        # Create new element
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
    """Handle when user saves edited text from the side panel.
    
    IMPORTANT: After user saves:
    1. We parse the text using star_txt_to_json and update state components
    2. We keep the user's text EXACTLY as saved (no regeneration)
    3. We evaluate to see which section needs improvement
    4. We ask questions to get more info, but NEVER modify the user's saved text
    """
    if VERBOSE:
        print("\n" + "="*50)
        print("💾 STAR TEXT SAVED BY USER")
        print("="*50)
        print(f"   Saved text length: {len(saved_text)}")
    
    # Store the saved text
    cl.user_session.set("saved_star_text", saved_text)
    
    # Parse the saved text using star_txt_to_json
    parsed = star_txt_to_json(saved_text)
    cl.user_session.set("saved_star_json", parsed)
    
    # Parse into components and update the state
    components = parse_star_text_to_components(saved_text)
    
    if VERBOSE:
        print("   📋 Parsed components:")
        print(f"      situation: {bool(components.get('situation'))} - '{components.get('situation', '')[:30]}...'")
        print(f"      task: {bool(components.get('task'))} - '{components.get('task', '')[:30]}...'")
        print(f"      action: {bool(components.get('action'))} - '{components.get('action', '')[:30]}...'")
        print(f"      result: {bool(components.get('result'))} - '{components.get('result', '')[:30]}...'")
    
    # Get current state and update with user edits
    state = cl.user_session.get("state")
    if state:
        # Update components - use parsed values (empty string if not present)
        state["situation"] = components.get("situation", "")
        state["task"] = components.get("task", "")
        state["action"] = components.get("action", "")
        state["result"] = components.get("result", "")
        
        # IMPORTANT: Keep the user's text EXACTLY as saved
        state["current_star_text"] = saved_text
        
        # Set flags to indicate user edited - skip regeneration
        state["user_edited"] = True
        state["skip_generate"] = True  # Skip generate node, go straight to evaluate
        
        # Clear pending question and section_to_improve for fresh evaluation
        state["pending_question"] = None
        state["section_to_improve"] = None
        
        # Set step to gather_info which will route to generate (which will skip to evaluate)
        state["current_step"] = "gather_info"
        
        if VERBOSE:
            print("   ✅ State updated with user edits")
            print("   🔒 User's text will be preserved (skip_generate=True)")
        
        # Save updated state
        cl.user_session.set("state", state)
    
    # DO NOT update EditableText here - keep user's exact text
    # The element already has the user's saved text
    
    # Notify user briefly
    await cl.Message(content="✅ Modifications sauvegardées! Évaluation en cours...").send()
    
    # Re-invoke the workflow to evaluate the saved text (without regenerating)
    if VERBOSE:
        print("\n⚙️ Re-invoking graph after user edit (will skip generation)...")
    
    # Run the graph with updated state
    # Add LangSmith metadata for the graph invocation
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
    
    # Update session state
    cl.user_session.set("state", result)
    
    # Get the element but DON'T change the text - it should stay as user saved it
    elem = cl.user_session.get("current_element")
    
    # Send response based on result
    if result.get("current_step") == "complete":
        final_message = f"""🎉 **Votre mission STAR est terminée !**

Votre texte a été validé.

🌟 Vous pouvez encore modifier le texte dans le panneau de droite si nécessaire."""
        
        if elem:
            await cl.Message(content=final_message, elements=[elem]).send()
        else:
            await cl.Message(content=final_message).send()
    
    elif result.get("pending_question"):
        # Build the message with the question
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
        # No pending question but not complete - might need more info
        await cl.Message(content="📝 Modifications prises en compte. Continuez à répondre aux questions pour améliorer votre texte STAR.").send()


@cl.on_chat_start
async def start():
    """Initialize the chat session"""
    
    # Generate a unique session ID for LangSmith tracing
    session_id = f"session-{uuid.uuid4()}"
    
    # Initialize LangGraph state
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

    # Store user and session IDs for LangSmith tracing
    cl.user_session.set("user_id", user_id)
    cl.user_session.set("session_id", session_id)
    
    # Store state in session
    cl.user_session.set("state", initial_state)
    
    # Initialize EditableText session variables
    cl.user_session.set("saved_star_text", None)
    cl.user_session.set("saved_star_json", None)
    cl.user_session.set("current_element", None)
    
    if VERBOSE:
        print(f"\n🚀 New session started")
        print(f"   user_id: {user_id}")
        print(f"   session_id: {session_id}")
    
    # Welcome message
    welcome = prompts.WELCOME_MESSAGE
    await cl.Message(content=welcome).send()


@cl.on_message
async def main(message: cl.Message):
    """Handle incoming messages.
    
    Routes SAVE_STAR_TEXT: messages to the save handler,
    all other messages go through the LangGraph workflow.
    """
    
    # Route save events to handler (from EditableText component)
    # The EditableText component automatically submits this and clears the input
    if message.content.startswith("SAVE_STAR_TEXT:"):
        saved_text = message.content.replace("SAVE_STAR_TEXT:", "", 1)
        if 'star_text_saved' in _event_handlers:
            await _event_handlers['star_text_saved'](saved_text)
        return
    
    # Handle regular messages through the workflow
    await handle_workflow_message(message)


async def handle_workflow_message(message: cl.Message):
    """Handle regular user messages through the LangGraph workflow"""
    
    state = cl.user_session.get("state")
    
    # Handle the conversation based on current step
    current_step = state.get("current_step", "gather_info")
    section_to_improve = state.get("section_to_improve")
    
    if VERBOSE:
        print("\n" + "="*50)
        print("📨 USER MESSAGE RECEIVED")
        print("="*50)
        print(f"   current_step: {current_step}")
        print(f"   section_to_improve: {section_to_improve}")
        print(f"   user_input: {message.content[:100]}...")
    
    # Check if we have previously saved text to use as context
    saved_star = cl.user_session.get("saved_star_text")
    if VERBOSE and saved_star:
        print(f"   📌 Previously saved text available in context")
    
    # Initial job description
    if not state.get("job_description"):
        state["job_description"] = message.content
        if VERBOSE:
            print("   → Stored as job_description")
        
    # If we're improving a specific section, update that section
    elif section_to_improve:
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
        # Clear section_to_improve after receiving the answer
        state["section_to_improve"] = None
        state["pending_question"] = None
        # Reset user_edited flag since we're adding new info
        state["user_edited"] = False
        state["skip_generate"] = False
        
    # Normal gathering flow (first time through each section)
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
    
    # Run synchronous graph.invoke in a thread pool to avoid blocking
    # Add LangSmith metadata for tracing
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
    
    # Update session state
    cl.user_session.set("state", result)
    
    # ALWAYS update EditableText if we have generated text
    elem = None
    if result.get("current_star_text"):
        elem = await update_editable_text(result["current_star_text"])
        if VERBOSE:
            print(f"   📝 EditableText updated with {len(result['current_star_text'])} chars")
    
    # Send response based on result
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
        # Build the message with the question
        section_name = result.get("section_to_improve", "").upper() if result.get("section_to_improve") else ""
        
        if section_name:
            question_msg = f"""🔍 **Section {section_name}:**

{result.get('pending_question')}"""
        else:
            question_msg = result.get('pending_question')
        
        # Always include the element if we have STAR text
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
    pass