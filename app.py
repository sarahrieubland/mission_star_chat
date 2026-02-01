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
- Question limits (per section and total)
- User exit mechanism

FIXES APPLIED:
- Fixed question count not incrementing properly
- Fixed infinite loop when agent re-asks same section
- Added state validation and persistence helpers
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
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o-mini")
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.0"))
VERBOSE = os.getenv("VERBOSE", "true").lower() == "true"

# LangSmith Hub configuration
LANGSMITH_HANDLE = os.getenv("LANGSMITH_HANDLE", "")
USE_HUB_PROMPTS = os.getenv("USE_HUB_PROMPTS", "true").lower() == "true"

# Question limits configuration
MAX_QUESTIONS_PER_SECTION = int(os.getenv("MAX_QUESTIONS_PER_SECTION", "4"))
MAX_TOTAL_QUESTIONS = int(os.getenv("MAX_TOTAL_QUESTIONS", "12"))

# --- LangSmith Configuration ---
os.environ.setdefault("LANGSMITH_TRACING", "true")

# Initialize LangSmith client
try:
    ls_client = Client()
    if VERBOSE:
        print("✅ LangSmith client initialized")
        print(f"   Project: {os.getenv('LANGSMITH_PROJECT', 'default')}")
        print(f"   Hub Handle: {LANGSMITH_HANDLE or 'Not set'}")
        print(f"📊 Question limits: {MAX_QUESTIONS_PER_SECTION} per section, {MAX_TOTAL_QUESTIONS} total")
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
    # Question tracking
    questions_per_section: dict
    total_questions_asked: int
    # User exit flag
    user_wants_to_exit: bool
    # Flag to indicate user just answered (prevents immediate re-asking)
    user_just_answered_for_section: Optional[str]
    # NEW: Track recently answered section to prevent loop
    _recently_answered_section: Optional[str]
    # NEW: Track consecutive questions on same section
    _consecutive_same_section_count: int
    _last_improved_section: Optional[str]


def validate_state(state: STARState) -> STARState:
    """Ensure all required tracking fields exist with valid defaults.
    
    This prevents KeyError issues and ensures state consistency.
    IMPORTANT: Only sets defaults for missing or None values, preserves existing values including 0.
    """
    defaults = {
        "questions_per_section": {"situation": 0, "task": 0, "action": 0, "result": 0},
        "total_questions_asked": 0,
        "_recently_answered_section": None,
        "user_just_answered_for_section": None,
        "_consecutive_same_section_count": 0,
        "_last_improved_section": None,
        "iteration_count": 0,
        "is_satisfactory": False,
        "user_edited": False,
        "skip_generate": False,
        "user_wants_to_exit": False,
    }
    
    for key, default in defaults.items():
        # IMPORTANT: Check if key exists in state. If it does, don't override it!
        # Only set default if key is truly missing (not in dict) or explicitly None
        if key not in state:
            state[key] = default
        elif state[key] is None and default is not None:
            # Only override None values if default is not None
            state[key] = default
        elif key == "questions_per_section":
            # Special handling for questions_per_section - ensure it's a dict and has all sections
            if not isinstance(state[key], dict):
                state[key] = default
            else:
                # Ensure all section keys exist
                for section in ["situation", "task", "action", "result"]:
                    if section not in state[key]:
                        state[key][section] = 0
    
    return state


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
    star_dict = {
        "situation": state.get('situation', '').strip(),
        "task": state.get('task', '').strip(),
        "action": state.get('action', '').strip(),
        "result": state.get('result', '').strip()
    }
    
    return star_json_to_txt(star_dict)


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
    """Node that gathers information about a specific STAR component.
    
    FIX: Now properly maintains question counters and prevents loops.
    """
    
    # FIX: Validate state at entry
    state = validate_state(state)
    
    if VERBOSE:
        print("\n🔄 GATHER_INFO_NODE")
        print(f"   🔍 STATE AT ENTRY:")
        print(f"      questions_per_section: {state.get('questions_per_section')}")
        print(f"      total_questions_asked: {state.get('total_questions_asked')}")
        print(f"   situation: '{state.get('situation', '')[:50]}' ({bool(state.get('situation'))})")
        print(f"   task: '{state.get('task', '')[:50]}' ({bool(state.get('task'))})")
        print(f"   action: '{state.get('action', '')[:50]}' ({bool(state.get('action'))})")
        print(f"   result: '{state.get('result', '')[:50]}' ({bool(state.get('result'))})")
        print(f"   section_to_improve: {state.get('section_to_improve')}")
        print(f"   user_edited: {state.get('user_edited', False)}")
        print(f"   _recently_answered: {state.get('_recently_answered_section')}")
    
    # FIX: Helper function to ensure counters are always preserved in returns
    # This needs to be a closure that captures the current state variable
    def create_return_state(current_state_ref, **updates):
        """Helper to ensure counters and tracking fields are always preserved"""
        return {
            **current_state_ref,  # Start with full current state
            "questions_per_section": current_state_ref.get("questions_per_section", {"situation": 0, "task": 0, "action": 0, "result": 0}),
            "total_questions_asked": current_state_ref.get("total_questions_asked", 0),
            "_recently_answered_section": current_state_ref.get("_recently_answered_section"),
            "_consecutive_same_section_count": current_state_ref.get("_consecutive_same_section_count", 0),
            "_last_improved_section": current_state_ref.get("_last_improved_section"),
            **updates  # Apply any specific updates
        }
    
    current_star_text = state.get("current_star_text", "")
    user_edited = state.get("user_edited", False)
    
    # Update STAR text if needed (but not if user just edited)
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
    
    # Helper function to check if we can ask more questions for a section
    def can_ask_question_for_section(section: str) -> bool:
        """Check if we can ask another question for this section."""
        section_count = state["questions_per_section"].get(section, 0)
        total_count = state.get("total_questions_asked", 0)
        
        if total_count >= MAX_TOTAL_QUESTIONS:
            if VERBOSE:
                print(f"   ⚠️ Cannot ask more questions: reached total limit ({MAX_TOTAL_QUESTIONS})")
            return False
        
        if section_count >= MAX_QUESTIONS_PER_SECTION:
            if VERBOSE:
                print(f"   ⚠️ Cannot ask more questions for {section}: reached section limit ({MAX_QUESTIONS_PER_SECTION})")
            return False
        
        return True
    
    # FIX: Helper function to increment question count - RETURNS updated state
    def increment_question_count(section: str, current_state: STARState) -> STARState:
        """Increment question counters for a section and RETURN updated state."""
        old_section_count = current_state["questions_per_section"].get(section, 0)
        old_total_count = current_state.get("total_questions_asked", 0)
        
        # Create new dicts to avoid mutation issues
        new_questions_per_section = dict(current_state["questions_per_section"])
        new_questions_per_section[section] = old_section_count + 1
        
        updated_state = {
            **current_state,
            "questions_per_section": new_questions_per_section,
            "total_questions_asked": old_total_count + 1
        }
        
        if VERBOSE:
            print(f"   📊 Question count updated:")
            print(f"      {section}: {old_section_count} → {new_questions_per_section[section]}")
            print(f"      total: {old_total_count} → {updated_state['total_questions_asked']}")
        
        return updated_state
    
    # FIX: Check if user just answered a question - clear flags and proceed
    user_just_answered = state.get("user_just_answered_for_section")
    
    if user_just_answered:
        # User just provided an answer - don't ask another question immediately
        # Clear the flags and let the workflow proceed (generate → evaluate)
        if VERBOSE:
            print(f"   ✅ User just answered for {user_just_answered.upper()}, proceeding to generate")
        
        return create_return_state(
            state,  # Pass current state reference
            pending_question=None,
            section_to_improve=None,
            user_just_answered_for_section=None,
            current_star_text=current_star_text,
            user_edited=False,
            _recently_answered_section=user_just_answered  # FIX: Mark as recently answered
        )
    
    # Check if evaluator wants us to improve a specific section
    if section_to_improve and can_ask_question_for_section(section_to_improve):
        prompt_map = {
            "situation": prompts.SITUATION_PROMPT,
            "task": prompts.TASK_PROMPT,
            "action": prompts.ACTION_PROMPT,
            "result": prompts.RESULT_PROMPT
        }
        
        prompt = prompt_map[section_to_improve].format(input=components_text)
        question = call_llm(prompt, run_name=f"ask_improve_{section_to_improve}")
        
        if VERBOSE:
            print(f"   📌 Asking to improve {section_to_improve.upper()}")
        
        # FIX: Capture returned state from increment
        state = increment_question_count(section_to_improve, state)
        
        return create_return_state(
            state,  # Pass UPDATED state reference
            pending_question=question,
            section_to_improve=section_to_improve,
            current_star_text=current_star_text,
            user_edited=False,
            user_just_answered_for_section=None
        )
    elif section_to_improve:
        # Can't ask more questions for this section
        if VERBOSE:
            print(f"   ⏭️ Cannot ask more questions for {section_to_improve}: limit reached")
        return create_return_state(
            state,  # Pass current state reference
            pending_question=None,
            section_to_improve=None,
            user_just_answered_for_section=None,
            current_star_text=current_star_text,
            user_edited=False
        )
    
    # Normal flow: determine which component needs info (first-time gathering)
    sections_to_check = [
        ("situation", state.get("situation")),
        ("task", state.get("task")),
        ("action", state.get("action")),
        ("result", state.get("result"))
    ]
    
    for section_name, section_value in sections_to_check:
        if not section_value and can_ask_question_for_section(section_name):
            prompt_map = {
                "situation": prompts.SITUATION_PROMPT.format(input=state.get('job_description', '')),
                "task": prompts.TASK_PROMPT.format(input=components_text),
                "action": prompts.ACTION_PROMPT.format(input=components_text),
                "result": prompts.RESULT_PROMPT.format(input=components_text)
            }
            
            question = call_llm(prompt_map[section_name], run_name=f"ask_{section_name}")
            
            if VERBOSE:
                print(f"   📌 Asking for {section_name.upper()} (first time)")
            
            # FIX: Capture returned state from increment
            state = increment_question_count(section_name, state)
            
            return create_return_state(
                state,  # Pass UPDATED state reference
                pending_question=question,
                section_to_improve=section_name,
                current_star_text=current_star_text,
                user_edited=False
            )
    
    if VERBOSE:
        print(f"   ✅ All components gathered or limits reached")
    
    return create_return_state(
        state,  # Pass current state reference
        pending_question=None,
        section_to_improve=None,
        current_star_text=current_star_text,
        user_edited=False
    )


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
    
    # First try using star_txt_to_json utility
    parsed = star_txt_to_json(llm_response)
    if parsed:
        # Map from parsed keys (which may vary) to standard keys
        extracted["situation"] = parsed.get("Situation", parsed.get("situation", "")).strip()
        extracted["task"] = parsed.get("Tâches", parsed.get("Tasks", parsed.get("task", ""))).strip()
        extracted["action"] = parsed.get("Actions", parsed.get("Action", parsed.get("action", ""))).strip()
        extracted["result"] = parsed.get("Résultats", parsed.get("Results", parsed.get("result", ""))).strip()
    else:
        # Fallback to regex parsing
        patterns = {
            "situation": [r"situation\s*:\s*", r"\*\*situation\*\*\s*:?\s*"],
            "task": [r"tâches?\s*:\s*", r"tasks?\s*:\s*", r"\*\*tâches?\*\*\s*:?\s*", r"\*\*tasks?\*\*\s*:?\s*"],
            "action": [r"actions?\s*:\s*", r"\*\*actions?\*\*\s*:?\s*"],
            "result": [r"résultats?\s*:\s*", r"results?\s*:\s*", r"\*\*résultats?\*\*\s*:?\s*", r"\*\*results?\*\*\s*:?\s*"]
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
                # Find the start of the next section header to exclude it
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
            
            # Clean up formatting
            content = re.sub(r'\*\*', '', content)
            content = content.strip()
            extracted[key] = content
    
    # Only use extracted content for sections that exist in state
    # This prevents duplication from re-extraction
    final = {
        "situation": extracted["situation"] if state.get("situation") else "",
        "task": extracted["task"] if state.get("task") else "",
        "action": extracted["action"] if state.get("action") else "",
        "result": extracted["result"] if state.get("result") else ""
    }
    
    # Fallback: if extraction failed but state has content, use state
    if not any(final.values()) and any([state.get("situation"), state.get("task"), state.get("action"), state.get("result")]):
        final = {
            "situation": state.get("situation", "").strip(),
            "task": state.get("task", "").strip(),
            "action": state.get("action", "").strip(),
            "result": state.get("result", "").strip()
        }
    
    # Build clean text WITHOUT duplicate headers using utility function
    text = star_json_to_txt(final)
    
    return text


def generate_node(state: STARState) -> STARState:
    """Node that generates the final STAR text when all components are gathered.
    
    FIX: Now properly preserves tracking fields.
    """
    
    # FIX: Validate state at entry
    state = validate_state(state)
    
    if state.get("skip_generate", False):
        if VERBOSE:
            print("\n🔄 GENERATE_NODE - SKIPPED (user edited text)")
        return {
            **state,
            "current_step": "evaluate",
            "skip_generate": False,
            # FIX: Preserve tracking fields
            "questions_per_section": state.get("questions_per_section", {}),
            "total_questions_asked": state.get("total_questions_asked", 0),
            "_recently_answered_section": state.get("_recently_answered_section"),
            "_consecutive_same_section_count": state.get("_consecutive_same_section_count", 0),
            "_last_improved_section": state.get("_last_improved_section"),
        }
    
    if VERBOSE:
        print("\n🔄 GENERATE_NODE")
        print(f"   🔍 STATE AT ENTRY:")
        print(f"      questions_per_section: {state.get('questions_per_section')}")
        print(f"      total_questions_asked: {state.get('total_questions_asked')}")
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
        "user_edited": False,
        # FIX: Preserve tracking fields
        "questions_per_section": state.get("questions_per_section", {}),
        "total_questions_asked": state.get("total_questions_asked", 0),
        "_recently_answered_section": state.get("_recently_answered_section"),
        "_consecutive_same_section_count": state.get("_consecutive_same_section_count", 0),
        "_last_improved_section": state.get("_last_improved_section"),
    }


def evaluate_node(state: STARState) -> STARState:
    """Node that autonomously evaluates the STAR text.
    
    FIX: Now prevents infinite loops by avoiding recently answered sections.
    """
    
    # FIX: Validate state at entry
    state = validate_state(state)
    
    if VERBOSE:
        print("\n🔄 EVALUATE_NODE")
        print(f"   🔍 STATE AT ENTRY:")
        print(f"      questions_per_section: {state.get('questions_per_section')}")
        print(f"      total_questions_asked: {state.get('total_questions_asked')}")
        print(f"   iteration_count: {state.get('iteration_count', 0)}")
        print(f"   user_edited: {state.get('user_edited', False)}")
        print(f"   _recently_answered: {state.get('_recently_answered_section')}")
        print(f"   _last_improved: {state.get('_last_improved_section')}")
        print(f"   _consecutive_count: {state.get('_consecutive_same_section_count', 0)}")
    
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
        return {
            **state,
            "current_step": "gather_info",
            "is_satisfactory": False,
            "section_to_improve": None,
            # FIX: Preserve tracking fields
            "questions_per_section": state.get("questions_per_section", {}),
            "total_questions_asked": state.get("total_questions_asked", 0),
            "_recently_answered_section": state.get("_recently_answered_section"),
            "_consecutive_same_section_count": state.get("_consecutive_same_section_count", 0),
            "_last_improved_section": state.get("_last_improved_section"),
        }
    
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
                "user_edited": False,
                # FIX: Preserve tracking fields
                "questions_per_section": state.get("questions_per_section", {}),
                "total_questions_asked": state.get("total_questions_asked", 0),
                "_recently_answered_section": None,  # Clear on completion
                "_consecutive_same_section_count": 0,
                "_last_improved_section": None,
            }
        
        # FIX: Check if we're about to ask about a section the user JUST answered
        recently_answered = state.get("_recently_answered_section")
        last_improved = state.get("_last_improved_section")
        consecutive_count = state.get("_consecutive_same_section_count", 0)
        
        # FIX: Prevent immediate re-asking of recently answered section
        if section_to_improve == recently_answered:
            if VERBOSE:
                print(f"   ⚠️ LOOP PREVENTION: Section {section_to_improve} was just answered")
                print(f"   🔄 Finding alternative section to improve")
            
            # Find sections with content that haven't been asked about recently
            sections_status = {
                "situation": bool(state.get("situation")),
                "task": bool(state.get("task")),
                "action": bool(state.get("action")),
                "result": bool(state.get("result"))
            }
            
            # Prioritize sections that exist and aren't the recently answered one
            other_sections = [s for s in ["situation", "task", "action", "result"]
                            if s != section_to_improve and sections_status[s]]
            
            # Check if we can still ask questions
            total_count = state.get("total_questions_asked", 0)
            
            if other_sections and total_count < MAX_TOTAL_QUESTIONS:
                # Find a section we can still ask about
                found_section = None
                for alt_section in other_sections:
                    section_count = state["questions_per_section"].get(alt_section, 0)
                    if section_count < MAX_QUESTIONS_PER_SECTION:
                        found_section = alt_section
                        break
                
                if found_section:
                    # Pick this alternative section
                    section_to_improve = found_section
                    
                    # Generate a new question for this different section
                    prompt_map = {
                        "situation": prompts.SITUATION_PROMPT,
                        "task": prompts.TASK_PROMPT,
                        "action": prompts.ACTION_PROMPT,
                        "result": prompts.RESULT_PROMPT
                    }
                    components, components_text = build_star_from_components(state)
                    prompt = prompt_map[section_to_improve].format(input=components_text)
                    question = call_llm(prompt, run_name=f"ask_improve_{section_to_improve}_alternative")
                    
                    if VERBOSE:
                        print(f"   ✅ Redirected to {section_to_improve.upper()} instead")
                else:
                    # No sections available
                    if VERBOSE:
                        print(f"   ✅ No other sections available, marking complete")
                    is_satisfactory = True
                    section_to_improve = None
                    question = None
            else:
                # No other sections to improve, mark as satisfactory
                if VERBOSE:
                    print(f"   ✅ No other sections to improve or limits reached, marking complete")
                is_satisfactory = True
                section_to_improve = None
                question = None
        
        # FIX: Track consecutive questions on same section to prevent over-focusing
        if section_to_improve == last_improved:
            consecutive_count += 1
            if VERBOSE:
                print(f"   📊 Consecutive questions on {section_to_improve}: {consecutive_count}")
            
            # If we've asked 2+ consecutive questions on same section, consider switching
            if consecutive_count >= 2:
                if VERBOSE:
                    print(f"   ⚠️ Too many consecutive questions on {section_to_improve}")
                
                # Try to find another section
                other_sections = [s for s in ["situation", "task", "action", "result"]
                                if s != section_to_improve and bool(state.get(s))]
                
                if other_sections:
                    old_section = section_to_improve
                    section_to_improve = other_sections[0]
                    consecutive_count = 0
                    
                    # Generate question for new section
                    prompt_map = {
                        "situation": prompts.SITUATION_PROMPT,
                        "task": prompts.TASK_PROMPT,
                        "action": prompts.ACTION_PROMPT,
                        "result": prompts.RESULT_PROMPT
                    }
                    components, components_text = build_star_from_components(state)
                    prompt = prompt_map[section_to_improve].format(input=components_text)
                    question = call_llm(prompt, run_name=f"ask_improve_{section_to_improve}_rotate")
                    
                    if VERBOSE:
                        print(f"   🔄 Rotated from {old_section} to {section_to_improve}")
        else:
            consecutive_count = 0
        
        # FIX: CRITICAL - Increment counter when evaluate_node decides to ask a question
        # This is necessary because gather_info_node won't increment when user_just_answered is set
        if not is_satisfactory and question and section_to_improve:
            # Check if we can ask this question
            section_count = state["questions_per_section"].get(section_to_improve, 0)
            total_count = state.get("total_questions_asked", 0)
            
            if total_count < MAX_TOTAL_QUESTIONS and section_count < MAX_QUESTIONS_PER_SECTION:
                # Increment counters for the question we're about to ask
                questions_per_section = dict(state.get("questions_per_section", {}))
                questions_per_section[section_to_improve] = section_count + 1
                total_count = total_count + 1
                
                if VERBOSE:
                    print(f"   📊 INCREMENT in evaluate (improvement question):")
                    print(f"      {section_to_improve}: {section_count} → {questions_per_section[section_to_improve]}")
                    print(f"      total: {state.get('total_questions_asked', 0)} → {total_count}")
            else:
                # Reached limit, mark as satisfactory instead
                if VERBOSE:
                    print(f"   ⚠️ Cannot ask question: limits reached")
                    print(f"      total: {total_count}/{MAX_TOTAL_QUESTIONS}")
                    print(f"      {section_to_improve}: {section_count}/{MAX_QUESTIONS_PER_SECTION}")
                is_satisfactory = True
                question = None
                section_to_improve = None
                questions_per_section = state.get("questions_per_section", {})
                total_count = state.get("total_questions_asked", 0)
        else:
            questions_per_section = state.get("questions_per_section", {})
            total_count = state.get("total_questions_asked", 0)
        
        # Return with updated tracking
        return {
            **state,
            "is_satisfactory": is_satisfactory,
            "current_step": "complete" if is_satisfactory else "gather_info",
            "pending_question": question,
            "section_to_improve": section_to_improve,
            "user_edited": False,
            # FIX: Use UPDATED counters
            "questions_per_section": questions_per_section,
            "total_questions_asked": total_count,
            "_recently_answered_section": None if is_satisfactory else recently_answered,
            "_consecutive_same_section_count": consecutive_count,
            "_last_improved_section": section_to_improve if not is_satisfactory else None,
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
            "user_edited": False,
            # FIX: Preserve tracking fields
            "questions_per_section": state.get("questions_per_section", {}),
            "total_questions_asked": state.get("total_questions_asked", 0),
            "_recently_answered_section": None,
            "_consecutive_same_section_count": 0,
            "_last_improved_section": None,
        }


def complete_node(state: STARState) -> STARState:
    """Final node when the STAR text is complete"""
    if VERBOSE:
        print("\n✅ COMPLETE_NODE")
        print(f"   Final iteration count: {state.get('iteration_count', 0)}")
        print(f"   Total questions asked: {state.get('total_questions_asked', 0)}")
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
        components["task"] = parsed.get("Tâches", "").strip()
        components["action"] = parsed.get("Actions", "").strip()
        components["result"] = parsed.get("Résultats", "").strip()
    
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
    """Handle when user saves edited text from the side panel.
    
    FIX: Now properly preserves question counters.
    """
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
        # FIX: Preserve all tracking fields when updating state
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
        
        # FIX: Explicitly preserve tracking fields (they should already exist but ensure they do)
        if "questions_per_section" not in state:
            state["questions_per_section"] = {"situation": 0, "task": 0, "action": 0, "result": 0}
        if "total_questions_asked" not in state:
            state["total_questions_asked"] = 0
        
        if VERBOSE:
            print("   ✅ State updated with user edits")
            print("   🔒 User's text will be preserved (skip_generate=True)")
            print(f"   📊 Counters preserved: {state.get('total_questions_asked', 0)} total questions")
        
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
        print(f"   total_questions_asked: {result.get('total_questions_asked', 0)}")
    
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
        questions_left = MAX_TOTAL_QUESTIONS - result.get("total_questions_asked", 0)
        
        if section_name:
            question_msg = f"""📝 Merci pour vos modifications!

🔍 **Pour améliorer la section {section_name}:**

{result.get('pending_question')}

💡 Tapez "terminer" à tout moment si le texte vous convient. ({questions_left} questions restantes)"""
        else:
            question_msg = f"""📝 Merci pour vos modifications!

{result.get('pending_question')}

💡 Tapez "terminer" à tout moment si le texte vous convient. ({questions_left} questions restantes)"""
        
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
        "skip_generate": False,
        "questions_per_section": {"situation": 0, "task": 0, "action": 0, "result": 0},
        "total_questions_asked": 0,
        "user_wants_to_exit": False,
        "user_just_answered_for_section": None,
        # FIX: Initialize new tracking fields
        "_recently_answered_section": None,
        "_consecutive_same_section_count": 0,
        "_last_improved_section": None,
    }

    cl.user_session.set("user_id", user_id)
    cl.user_session.set("session_id", session_id)
    cl.user_session.set("state", initial_state)
    cl.user_session.set("saved_star_text", None)
    cl.user_session.set("saved_star_json", None)
    cl.user_session.set("current_element", None)
    cl.user_session.set("extraction_done", False)
    
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
    
    FIX: Now properly tracks question counts and prevents loops.
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
        print(f"   total_questions_asked: {state.get('total_questions_asked', 0)}")
    
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
        
        extraction_summary += """

📝 Le texte STAR est affiché dans le panneau de droite.

💬 Répondez simplement "ok" ou "continue" pour commencer l'amélioration,
   ou tapez "terminer" si le texte vous convient déjà."""
        
        if elem:
            await cl.Message(content=extraction_summary, elements=[elem]).send()
        else:
            await cl.Message(content=extraction_summary).send()
        
        return  # Don't invoke graph yet, wait for user response
    
    # --- SUBSEQUENT MESSAGES: LangGraph Workflow ---
    
    # Check if user wants to exit
    user_message_lower = message.content.lower().strip()
    exit_keywords = ["terminer", "termine", "exit", "quit", "done", "c'est bon", "ça suffit", "ca suffit"]
    
    if any(keyword in user_message_lower for keyword in exit_keywords):
        if VERBOSE:
            print(f"   🚪 User wants to exit")
        
        state["user_wants_to_exit"] = True
        state["is_satisfactory"] = True
        state["current_step"] = "complete"
        cl.user_session.set("state", state)
        
        elem = cl.user_session.get("current_element")
        final_message = f"""✅ **Mission STAR terminée à votre demande !**

Voici votre texte STAR final :

---

{state.get('current_star_text', '')}

---

🌟 Vous pouvez toujours modifier le texte dans le panneau de droite si nécessaire."""
        
        if elem:
            await cl.Message(content=final_message, elements=[elem]).send()
        else:
            await cl.Message(content=final_message).send()
        
        return
    
    # Check if user confirms to start after extraction
    confirmation_keywords = ["ok", "oui", "yes", "continue", "continuer", "commencer", "commence", "go", "allez-y", "allez"]
    
    # If no components gathered yet and user sent confirmation, start workflow
    if not any([state.get("situation"), state.get("task"), state.get("action"), state.get("result")]) or \
       (state.get("total_questions_asked", 0) == 0 and any(keyword in user_message_lower for keyword in confirmation_keywords)):
        if any(keyword in user_message_lower for keyword in confirmation_keywords):
            if VERBOSE:
                print(f"   ✅ User confirmed to start improvement")
            
            # Now invoke the graph for the first time
            result = await asyncio.to_thread(
                graph.invoke,
                state,
                config={
                    "run_name": "star_workflow_start",
                    "metadata": {
                        "user_id": cl.user_session.get("user_id", user_id),
                        "session_id": cl.user_session.get("session_id", "unknown"),
                        "trigger": "user_confirmation",
                    }
                }
            )
            
            cl.user_session.set("state", result)
            
            # Update EditableText if text changed
            elem = cl.user_session.get("current_element")
            if result.get("current_star_text") and result.get("current_star_text") != state.get("current_star_text"):
                elem = await update_editable_text(result["current_star_text"])
            
            # Send the first question from the workflow
            if result.get("pending_question"):
                section_name = result.get("section_to_improve", "").upper() if result.get("section_to_improve") else ""
                questions_left = MAX_TOTAL_QUESTIONS - result.get("total_questions_asked", 0)
                
                if section_name:
                    question_msg = f"""🔍 **Section {section_name}:**

{result.get('pending_question')}

💡 Tapez "terminer" à tout moment si le texte vous convient. ({questions_left} questions restantes)"""
                else:
                    question_msg = f"""{result.get('pending_question')}

💡 Tapez "terminer" à tout moment si le texte vous convient. ({questions_left} questions restantes)"""
                
                if elem:
                    await cl.Message(content=question_msg, elements=[elem]).send()
                else:
                    await cl.Message(content=question_msg).send()
            
            return
    
    # FIX: Normal workflow processing - update state with user's answer
    saved_star = cl.user_session.get("saved_star_text")
    if VERBOSE and saved_star:
        print(f"   📌 Previously saved text available in context")
    
    if section_to_improve:
        # User is answering a question about a specific section
        # FIX: Preserve existing counters when updating
        questions_per_section = state.get("questions_per_section", {"situation": 0, "task": 0, "action": 0, "result": 0})
        total_questions = state.get("total_questions_asked", 0)
        
        if section_to_improve == "situation":
            state["situation"] = state.get("situation", "") + " " + message.content
            if VERBOSE:
                print(f"   → Appended to SITUATION")
                print(f"   Current situation text: {state['situation'][:100]}...")
        elif section_to_improve == "task":
            state["task"] = state.get("task", "") + " " + message.content
            if VERBOSE:
                print(f"   → Appended to TASK")
                print(f"   Current task text: {state['task'][:100]}...")
        elif section_to_improve == "action":
            state["action"] = state.get("action", "") + " " + message.content
            if VERBOSE:
                print(f"   → Appended to ACTION")
                print(f"   Current action text: {state['action'][:100]}...")
        elif section_to_improve == "result":
            state["result"] = state.get("result", "") + " " + message.content
            if VERBOSE:
                print(f"   → Appended to RESULT")
                print(f"   Current result text: {state['result'][:100]}...")
        
        # FIX: Set flag to indicate user just answered for this section
        state["user_just_answered_for_section"] = section_to_improve
        state["pending_question"] = None
        state["user_edited"] = False
        state["skip_generate"] = False
        
        # FIX: CRITICAL - Explicitly preserve counters
        state["questions_per_section"] = questions_per_section
        state["total_questions_asked"] = total_questions
        
        if VERBOSE:
            print(f"   📊 Preserved counters:")
            print(f"      questions_per_section: {state['questions_per_section']}")
            print(f"      total_questions_asked: {state['total_questions_asked']}")
        
        # CRITICAL: Save updated state to session BEFORE invoking graph
        cl.user_session.set("state", state)
        
    elif current_step == "gather_info":
        # FIX: Preserve existing counters when updating
        questions_per_section = state.get("questions_per_section", {"situation": 0, "task": 0, "action": 0, "result": 0})
        total_questions = state.get("total_questions_asked", 0)
        
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
        
        # FIX: CRITICAL - Explicitly preserve counters
        state["questions_per_section"] = questions_per_section
        state["total_questions_asked"] = total_questions
        
        if VERBOSE:
            print(f"   📊 Preserved counters:")
            print(f"      questions_per_section: {state['questions_per_section']}")
            print(f"      total_questions_asked: {state['total_questions_asked']}")
        
        # CRITICAL: Save updated state to session BEFORE invoking graph
        cl.user_session.set("state", state)
    
    if VERBOSE:
        print("\n⚙️ Invoking graph...")
        print(f"   State before invoke - total_questions: {state.get('total_questions_asked', 0)}")
    
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
        print(f"   total_questions_asked: {result.get('total_questions_asked', 0)}")
        print(f"   questions_per_section: {result.get('questions_per_section', {})}")
    
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
        
        questions_left = MAX_TOTAL_QUESTIONS - result.get("total_questions_asked", 0)
        
        if section_name:
            question_msg = f"""🔍 **Section {section_name}:**

{result.get('pending_question')}

💡 Tapez "terminer" à tout moment si le texte vous convient. ({questions_left} questions restantes)"""
        else:
            question_msg = f"""{result.get('pending_question')}

💡 Tapez "terminer" à tout moment si le texte vous convient. ({questions_left} questions restantes)"""
        
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