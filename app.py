"""
STAR Text Generator - Agentic Workflow with LangGraph and Chainlit

REDESIGNED GENERATION APPROACH:
Instead of appending user answers to state, we now pass 3 inputs to generation:
1. Existing STAR text (from previous iteration)
2. Question that was asked
3. User's answer to that question

This allows the LLM to intelligently integrate new information into existing text.

Features:
- LangGraph agentic workflow for gathering and improving STAR components
- Editable side panel showing the current STAR text at EVERY step
- Context-aware generation (existing text + question + answer)
- LangSmith integration for tracing and monitoring
- Question limits (per section and total)
- Configurable temperatures for different tasks
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

# Different temperatures for different tasks
TEMP_QUESTIONS = float(os.getenv("TEMP_QUESTIONS", "0.3"))  # Precise questions
TEMP_EVALUATION = float(os.getenv("TEMP_EVALUATION", "0.2"))  # Consistent evaluation
TEMP_GENERATION = float(os.getenv("TEMP_GENERATION", "0.7"))  # Creative integration
TEMP_EXTRACTION = float(os.getenv("TEMP_EXTRACTION", "0.0"))  # Exact extraction

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
        print(f"🌡️  Temperatures: Questions={TEMP_QUESTIONS}, Eval={TEMP_EVALUATION}, Gen={TEMP_GENERATION}, Extract={TEMP_EXTRACTION}")
except Exception as e:
    ls_client = None
    if VERBOSE:
        print(f"⚠️ LangSmith client initialization failed: {e}")

# Generate a unique user ID for this session
user_id = f"user-{uuid.uuid4()}"

# Registry for custom event handlers
_event_handlers = {}

# Initialize prompt manager
prompts = PromptManager(handle=LANGSMITH_HANDLE, use_hub=USE_HUB_PROMPTS, client=ls_client)


class STARState(TypedDict):
    """State that persists across the workflow for the agent"""
    job_description: str
    # NEW: Store raw user responses separately
    situation_raw: str
    task_raw: str
    action_raw: str
    result_raw: str
    # Current polished STAR text
    current_star_text: str
    # Context for generation
    last_question_asked: Optional[str]
    last_answer_given: Optional[str]
    last_section_improved: Optional[str]
    # Workflow control
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
    # Loop prevention
    user_just_answered_for_section: Optional[str]
    _recently_answered_section: Optional[str]
    _consecutive_same_section_count: int
    _last_improved_section: Optional[str]


def validate_state(state: STARState) -> STARState:
    """Ensure all required tracking fields exist with valid defaults."""
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
        "last_question_asked": None,
        "last_answer_given": None,
        "last_section_improved": None,
        # NEW: Initialize raw fields
        "situation_raw": "",
        "task_raw": "",
        "action_raw": "",
        "result_raw": "",
    }
    
    for key, default in defaults.items():
        if key not in state:
            state[key] = default
        elif state[key] is None and default is not None:
            state[key] = default
        elif key == "questions_per_section":
            if not isinstance(state[key], dict):
                state[key] = default
            else:
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


def call_llm(prompt: str, system: str = None, run_name: str = None, temperature: float = None) -> str:
    """Make an LLM call with the given prompt and temperature."""
    if system is None:
        system = prompts.AGENT_SYSTEM_PROMPT
    
    # Use provided temperature or default
    if temperature is None:
        temperature = TEMP_QUESTIONS
    
    # Create LLM with specific temperature
    llm = ChatOpenAI(
        model=MODEL_NAME,
        temperature=temperature,
    )
    
    messages = [
        SystemMessage(content=system),
        HumanMessage(content=prompt)
    ]
    
    if VERBOSE:
        print("\n" + "="*50)
        print(f"📤 LLM CALL{f' ({run_name})' if run_name else ''}")
        print("="*50)
        print(f"🤖 Model: {MODEL_NAME}")
        print(f"🌡️  Temperature: {temperature}")
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
                "temperature": temperature,
            }
        }
    )
    
    if VERBOSE:
        print(f"📥 Response: {response.content[:200]}..." if len(response.content) > 200 else f"📥 Response: {response.content}")
        print("="*50 + "\n")
    
    return response.content


def extract_star_from_text(input_text: str) -> dict:
    """Extract STAR components from user's initial job description.
    
    The LLM handles all formats (plain text, markdown, formatted, etc.)
    """
    if VERBOSE:
        print("\n" + "="*50)
        print("🔍 INITIAL STAR EXTRACTION")
        print("="*50)
        print(f"   Input: {input_text[:200]}...")
    
    # Simply send the text to the LLM - it handles all formats
    prompt = prompts.EXTRACTION_PROMPT.format(input_text=input_text)
    response = call_llm(prompt, system="", run_name="initial_star_extraction", temperature=TEMP_EXTRACTION)
    
    try:
        cleaned = response.strip()
        if cleaned.startswith("```"):
            parts = cleaned.split("```")
            if len(parts) >= 2:
                cleaned = parts[1]
                if cleaned.startswith("json"):
                    cleaned = cleaned[4:]
        cleaned = cleaned.strip()
        
        extracted = json.loads(cleaned)
        
        result = {
            "situation": extracted.get("situation", "").strip(),
            "task": extracted.get("task", "").strip(),
            "action": extracted.get("action", "").strip(),
            "result": extracted.get("result", "").strip()
        }
        
        if VERBOSE:
            print("   ✅ Extraction successful:")
            for key, val in result.items():
                if val:
                    print(f"      {key}: {len(val)} chars - '{val[:50]}...'")
                else:
                    print(f"      {key}: (empty)")
            print("="*50 + "\n")
        
        return result
        
    except (json.JSONDecodeError, KeyError) as e:
        if VERBOSE:
            print(f"   ⚠️ Failed to parse extraction response: {e}")
            print(f"   Raw response: {response[:300]}...")
            print("="*50 + "\n")
        
        return {"situation": "", "task": "", "action": "", "result": ""}


def format_star_text_from_state(state: STARState) -> str:
    """Simple formatting of STAR text from state (no LLM)."""
    star_dict = {
        "situation": state.get('situation_raw', '').strip(),
        "task": state.get('task_raw', '').strip(),
        "action": state.get('action_raw', '').strip(),
        "result": state.get('result_raw', '').strip()
    }
    return star_json_to_txt(star_dict)


def generate_star_text(state: STARState) -> str:
    """Generate STAR text using context-aware approach.
    
    NEW APPROACH:
    - If first generation: use raw components
    - If improving: use existing_text + question + answer
    
    This allows LLM to intelligently integrate new information.
    """
    
    # Check if we have context (question + answer)
    has_context = (
        state.get("last_question_asked") and 
        state.get("last_answer_given") and
        state.get("last_section_improved")
    )
    
    if has_context and state.get("current_star_text"):
        # IMPROVEMENT MODE: Use existing text + question + answer
        if VERBOSE:
            print("\n" + "="*50)
            print("   📝 GENERATING IMPROVED STAR TEXT (Context-Aware)")
            print("="*50)
            print(f"   Section being improved: {state['last_section_improved'].upper()}")
            print(f"   Question asked: {state['last_question_asked'][:100]}...")
            print(f"   Answer given: {state['last_answer_given'][:100]}...")
            print(f"   Existing text length: {len(state['current_star_text'])} chars")
            print("-"*50)
        
        prompt = prompts.GENERATE_STAR_PROMPT.format(
            existing_text=state["current_star_text"],
            question=state["last_question_asked"],
            answer=state["last_answer_given"],
            section=state["last_section_improved"]
        )
        
    else:
        # INITIAL MODE: Use raw components
        if VERBOSE:
            print("\n" + "="*50)
            print("   📝 GENERATING INITIAL STAR TEXT (From Raw Components)")
            print("="*50)
        
        # Build raw components text
        components = []
        if state.get("situation_raw"):
            components.append(f"SITUATION: {state['situation_raw']}")
        if state.get("task_raw"):
            components.append(f"TÂCHE: {state['task_raw']}")
        if state.get("action_raw"):
            components.append(f"ACTION: {state['action_raw']}")
        if state.get("result_raw"):
            components.append(f"RÉSULTAT: {state['result_raw']}")
        
        if not components:
            return ""
        
        components_text = "\n".join(components)
        
        if VERBOSE:
            print(f"   Components: {len(components)}")
            print(f"   Input length: {len(components_text)} chars")
            print("-"*50)
        
        # For initial generation, use a simpler prompt format
        # (You can create a separate INITIAL_GENERATE_PROMPT if needed)
        prompt = f"""À partir des éléments bruts suivants, créez une description STAR professionnelle et bien structurée.

Éléments fournis:
{components_text}

Créez une description au format:
Situation:
[2-3 phrases]

Tâches:
[2-3 phrases]

Actions:
[Puces]

Résultats:
[Puces]"""
    
    if VERBOSE:
        print("   Calling LLM for generation...")
        print("-"*50)
    
    llm_response = call_llm(
        prompt, 
        run_name="generate_star_text",
        temperature=TEMP_GENERATION  # Use creative temperature
    )
    
    # Parse the response back to standard format
    formatted_text = reformat_llm_response_to_standard(llm_response, state)
    
    if VERBOSE:
        print(f"   ✅ Generated text length: {len(formatted_text)} chars")
        print("="*50 + "\n")
    
    return formatted_text


def reformat_llm_response_to_standard(llm_response: str, state: STARState) -> str:
    """Parse LLM response into standard format."""
    import re
    
    # Try using the utility parser first
    parsed = star_txt_to_json(llm_response)
    
    if parsed:
        result = {
            "situation": parsed.get("Situation", "").strip(),
            "task": parsed.get("Tâches", "").strip(),
            "action": parsed.get("Actions", "").strip(),
            "result": parsed.get("Résultats", "").strip()
        }
        return star_json_to_txt(result)
    
    # Fallback: return the LLM response as-is if it's already formatted
    # The LLM should return properly formatted text
    if VERBOSE:
        print("   ⚠️ Could not parse with star_txt_to_json, returning LLM response as-is")
    
    return llm_response.strip()


# === WORKFLOW NODES ===

def gather_info_node(state: STARState) -> STARState:
    """Gather information or prepare for generation."""
    
    state = validate_state(state)
    
    if VERBOSE:
        print("\n🔄 GATHER_INFO_NODE")
        print(f"   🔍 STATE AT ENTRY:")
        print(f"      questions_per_section: {state.get('questions_per_section')}")
        print(f"      total_questions_asked: {state.get('total_questions_asked')}")
        print(f"      section_to_improve: {state.get('section_to_improve')}")
        print(f"      user_just_answered: {state.get('user_just_answered_for_section')}")
    
    def create_return_state(current_state_ref, **updates):
        """Helper to ensure counters are preserved."""
        return {
            **current_state_ref,
            "questions_per_section": current_state_ref.get("questions_per_section", {}),
            "total_questions_asked": current_state_ref.get("total_questions_asked", 0),
            "_recently_answered_section": current_state_ref.get("_recently_answered_section"),
            "_consecutive_same_section_count": current_state_ref.get("_consecutive_same_section_count", 0),
            "_last_improved_section": current_state_ref.get("_last_improved_section"),
            **updates
        }
    
    # Update current text with simple formatting (no LLM)
    current_star_text = state.get("current_star_text", "")
    if not state.get("user_edited"):
        has_any = any([
            state.get("situation_raw"),
            state.get("task_raw"),
            state.get("action_raw"),
            state.get("result_raw")
        ])
        if has_any:
            current_star_text = format_star_text_from_state(state)
    
    section_to_improve = state.get("section_to_improve")
    
    # Helper functions
    def can_ask_question_for_section(section: str) -> bool:
        section_count = state["questions_per_section"].get(section, 0)
        total_count = state.get("total_questions_asked", 0)
        return (total_count < MAX_TOTAL_QUESTIONS and 
                section_count < MAX_QUESTIONS_PER_SECTION)
    
    def increment_question_count(section: str, current_state: STARState) -> STARState:
        old_section_count = current_state["questions_per_section"].get(section, 0)
        old_total_count = current_state.get("total_questions_asked", 0)
        
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
    
    user_just_answered = state.get("user_just_answered_for_section")
    
    if user_just_answered:
        if VERBOSE:
            print(f"   ✅ User just answered for {user_just_answered.upper()}, proceeding to generate")
        
        return create_return_state(
            state,
            pending_question=None,
            section_to_improve=None,
            user_just_answered_for_section=None,
            current_star_text=current_star_text,
            user_edited=False,
            _recently_answered_section=user_just_answered
        )
    
    # Check if evaluator wants us to improve a specific section
    if section_to_improve and can_ask_question_for_section(section_to_improve):
        prompt_map = {
            "situation": prompts.SITUATION_PROMPT,
            "task": prompts.TASK_PROMPT,
            "action": prompts.ACTION_PROMPT,
            "result": prompts.RESULT_PROMPT
        }
        
        # Build context for question
        components, components_text = build_star_from_components({
            'situation': state.get('situation_raw'),
            'task': state.get('task_raw'),
            'action': state.get('action_raw'),
            'result': state.get('result_raw')
        })
        
        prompt = prompt_map[section_to_improve].format(input=components_text)
        question = call_llm(prompt, run_name=f"ask_improve_{section_to_improve}", temperature=TEMP_QUESTIONS)
        
        if VERBOSE:
            print(f"   📌 Asking to improve {section_to_improve.upper()}")
        
        state = increment_question_count(section_to_improve, state)
        
        return create_return_state(
            state,
            pending_question=question,
            section_to_improve=section_to_improve,
            current_star_text=current_star_text,
            user_edited=False,
            user_just_answered_for_section=None
        )
    
    # First-time gathering
    sections_to_check = [
        ("situation", state.get("situation_raw")),
        ("task", state.get("task_raw")),
        ("action", state.get("action_raw")),
        ("result", state.get("result_raw"))
    ]
    
    for section_name, section_value in sections_to_check:
        if not section_value and can_ask_question_for_section(section_name):
            prompt_map = {
                "situation": prompts.SITUATION_PROMPT.format(input=state.get('job_description', '')),
                "task": prompts.TASK_PROMPT.format(input=""),
                "action": prompts.ACTION_PROMPT.format(input=""),
                "result": prompts.RESULT_PROMPT.format(input="")
            }
            
            question = call_llm(prompt_map[section_name], run_name=f"ask_{section_name}", temperature=TEMP_QUESTIONS)
            
            if VERBOSE:
                print(f"   📌 Asking for {section_name.upper()} (first time)")
            
            state = increment_question_count(section_name, state)
            
            return create_return_state(
                state,
                pending_question=question,
                section_to_improve=section_name,
                current_star_text=current_star_text,
                user_edited=False
            )
    
    if VERBOSE:
        print(f"   ✅ All components gathered or limits reached")
    
    return create_return_state(
        state,
        pending_question=None,
        section_to_improve=None,
        current_star_text=current_star_text,
        user_edited=False
    )


def generate_node(state: STARState) -> STARState:
    """Generate improved STAR text."""
    
    state = validate_state(state)
    
    if state.get("skip_generate", False):
        if VERBOSE:
            print("\n🔄 GENERATE_NODE - SKIPPED (user edited text)")
        return {
            **state,
            "current_step": "evaluate",
            "skip_generate": False,
            "questions_per_section": state.get("questions_per_section", {}),
            "total_questions_asked": state.get("total_questions_asked", 0),
        }
    
    if VERBOSE:
        print("\n🔄 GENERATE_NODE")
        print(f"   🔍 STATE AT ENTRY:")
        print(f"      questions_per_section: {state.get('questions_per_section')}")
        print(f"      total_questions_asked: {state.get('total_questions_asked')}")
        print(f"      last_section_improved: {state.get('last_section_improved')}")
    
    # Store old text for comparison
    old_star_text = state.get("current_star_text", "")
    
    star_text = generate_star_text(state)
    
    if VERBOSE:
        print("📥 GENERATED STAR TEXT:")
        print(star_text)
        print("="*50 + "\n")
    
    # FIX: Parse the generated text back to raw fields
    # This ensures improvements persist for next iteration
    parsed_components = parse_star_text_to_components(star_text)
    
    if VERBOSE:
        print("   💾 Persisting generated text to raw fields...")
        for section, content in parsed_components.items():
            if content:
                old_len = len(state.get(f"{section}_raw", ""))
                new_len = len(content)
                print(f"      {section}_raw: {old_len} → {new_len} chars")
    
    # Validation: check what changed
    if VERBOSE and state.get("last_section_improved"):
        old_components = parse_star_text_to_components(old_star_text) if old_star_text else {}
        section_improved = state.get("last_section_improved")
        
        print("\n   🔍 VALIDATION:")
        for section in ["situation", "task", "action", "result"]:
            old_content = old_components.get(section, "")
            new_content = parsed_components.get(section, "")
            
            if section == section_improved:
                if old_content != new_content:
                    print(f"      ✅ {section.upper()}: IMPROVED")
                else:
                    print(f"      ⚠️  {section.upper()}: NOT CHANGED")
            else:
                if old_content != new_content:
                    print(f"      ⚠️  {section.upper()}: CHANGED (should be preserved)")
                else:
                    print(f"      ✅ {section.upper()}: PRESERVED")
    
    return {
        **state,
        "current_star_text": star_text,
        # FIX: Update raw fields with generated content
        "situation_raw": parsed_components.get("situation", state.get("situation_raw", "")),
        "task_raw": parsed_components.get("task", state.get("task_raw", "")),
        "action_raw": parsed_components.get("action", state.get("action_raw", "")),
        "result_raw": parsed_components.get("result", state.get("result_raw", "")),
        "current_step": "evaluate",
        "iteration_count": state.get("iteration_count", 0) + 1,
        "section_to_improve": None,
        "user_edited": False,
        "questions_per_section": state.get("questions_per_section", {}),
        "total_questions_asked": state.get("total_questions_asked", 0),
        # Clear context after generation
        "last_question_asked": None,
        "last_answer_given": None,
        "last_section_improved": None,
    }


def evaluate_node(state: STARState) -> STARState:
    """Evaluate the STAR text."""
    
    state = validate_state(state)
    
    if VERBOSE:
        print("\n🔄 EVALUATE_NODE")
        print(f"   iteration_count: {state.get('iteration_count', 0)}")
    
    all_components = all([
        state.get("situation_raw"),
        state.get("task_raw"),
        state.get("action_raw"),
        state.get("result_raw")
    ])
    
    if not all_components:
        return {
            **state,
            "current_step": "gather_info",
            "is_satisfactory": False,
            "questions_per_section": state.get("questions_per_section", {}),
            "total_questions_asked": state.get("total_questions_asked", 0),
        }
    
    prompt = prompts.EVALUATE_PROMPT.format(input=state['current_star_text'])
    response = call_llm(prompt, run_name="evaluate_star_text", temperature=TEMP_EVALUATION)
    
    try:
        cleaned = response.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.split("```")[1]
            if cleaned.startswith("json"):
                cleaned = cleaned[4:]
        cleaned = cleaned.strip()
        
        evaluation = json.loads(cleaned)
        is_satisfactory = evaluation.get("is_satisfactory", False)
        section_to_improve = evaluation.get("section_to_improve")
        question = evaluation.get("question")
        
        if VERBOSE:
            print(f"   ✅ is_satisfactory: {is_satisfactory}")
            print(f"   📌 section_to_improve: {section_to_improve}")
        
        if is_satisfactory:
            return {
                **state,
                "is_satisfactory": True,
                "current_step": "complete",
                "pending_question": None,
                "section_to_improve": None,
                "questions_per_section": state.get("questions_per_section", {}),
                "total_questions_asked": state.get("total_questions_asked", 0),
            }
        
        # NOTE: Loop prevention removed - it's OK to ask same section 2x in a row
        # We only check the per-section and total limits
        
        # Increment counter
        if not is_satisfactory and question and section_to_improve:
            section_count = state["questions_per_section"].get(section_to_improve, 0)
            total_count = state.get("total_questions_asked", 0)
            
            if total_count < MAX_TOTAL_QUESTIONS and section_count < MAX_QUESTIONS_PER_SECTION:
                questions_per_section = dict(state.get("questions_per_section", {}))
                questions_per_section[section_to_improve] = section_count + 1
                total_count = total_count + 1
                
                if VERBOSE:
                    print(f"   📊 INCREMENT: {section_to_improve}: {section_count} → {questions_per_section[section_to_improve]}")
                    print(f"      total: {state.get('total_questions_asked', 0)} → {total_count}")
            else:
                is_satisfactory = True
                question = None
                section_to_improve = None
                questions_per_section = state.get("questions_per_section", {})
                total_count = state.get("total_questions_asked", 0)
        else:
            questions_per_section = state.get("questions_per_section", {})
            total_count = state.get("total_questions_asked", 0)
        
        return {
            **state,
            "is_satisfactory": is_satisfactory,
            "current_step": "complete" if is_satisfactory else "gather_info",
            "pending_question": question,
            "section_to_improve": section_to_improve,
            "questions_per_section": questions_per_section,
            "total_questions_asked": total_count,
        }
    
    except (json.JSONDecodeError, KeyError) as e:
        if VERBOSE:
            print(f"⚠️ Failed to parse evaluation: {e}")
        return {
            **state,
            "is_satisfactory": True,
            "current_step": "complete",
            "questions_per_section": state.get("questions_per_section", {}),
            "total_questions_asked": state.get("total_questions_asked", 0),
        }


def complete_node(state: STARState) -> STARState:
    """Final node."""
    if VERBOSE:
        print("\n✅ COMPLETE_NODE")
        print(f"   Total questions: {state.get('total_questions_asked', 0)}")
    return {**state, "current_step": "complete"}


# === ROUTERS ===

def after_gather_router(state: STARState) -> str:
    all_components = all([
        state.get("situation_raw"),
        state.get("task_raw"),
        state.get("action_raw"),
        state.get("result_raw")
    ])
    
    if state.get("pending_question"):
        return END
    elif all_components:
        return "generate"
    else:
        return END


def after_evaluate_router(state: STARState) -> str:
    if state.get("is_satisfactory"):
        return "complete"
    elif state.get("pending_question"):
        return END
    else:
        return "gather_info"


# === BUILD GRAPH ===

def create_star_graph():
    workflow = StateGraph(STARState)
    
    workflow.add_node("gather_info", gather_info_node)
    workflow.add_node("generate", generate_node)
    workflow.add_node("evaluate", evaluate_node)
    workflow.add_node("complete", complete_node)
    
    workflow.set_entry_point("gather_info")
    
    workflow.add_conditional_edges(
        "gather_info",
        after_gather_router,
        {"generate": "generate", END: END}
    )
    
    workflow.add_edge("generate", "evaluate")
    
    workflow.add_conditional_edges(
        "evaluate",
        after_evaluate_router,
        {"complete": "complete", "gather_info": "gather_info", END: END}
    )
    
    workflow.add_edge("complete", END)
    
    return workflow.compile()


graph = create_star_graph()


# === CHAINLIT HANDLERS ===

def parse_star_text_to_components(star_text: str) -> dict:
    """Parse STAR text back to components."""
    parsed = star_txt_to_json(star_text)
    
    if parsed:
        return {
            "situation": parsed.get("Situation", "").strip(),
            "task": parsed.get("Tâches", "").strip(),
            "action": parsed.get("Actions", "").strip(),
            "result": parsed.get("Résultats", "").strip()
        }
    return {"situation": "", "task": "", "action": "", "result": ""}


async def update_editable_text(star_text: str):
    """Update EditableText element."""
    elem = cl.user_session.get("current_element")
    
    if elem:
        elem.props["initial"] = star_text
        await elem.update()
    else:
        elem = cl.CustomElement(
            name="EditableText",
            display="inline",
            props={"initial": star_text, "keepVisible": True}
        )
        cl.user_session.set("current_element", elem)
    
    return elem


@on_star_text_saved
async def handle_save_action(saved_text: str):
    """Handle user edits and continue workflow."""
    if VERBOSE:
        print("\n💾 STAR TEXT SAVED BY USER")
        print(f"   Saved text length: {len(saved_text)} chars")
    
    cl.user_session.set("saved_star_text", saved_text)
    
    components = parse_star_text_to_components(saved_text)
    
    state = cl.user_session.get("state")
    if state:
        # Check if there was a pending question - if so, mark it as answered
        section_being_answered = state.get("section_to_improve")
        had_pending_question = state.get("pending_question") is not None
        
        if VERBOSE and section_being_answered:
            print(f"   📝 User saving answer for section: {section_being_answered.upper()}")
            print(f"   Previous question: {state.get('pending_question', '')[:100]}...")
        
        # Update raw fields with user's edits
        state["situation_raw"] = components.get("situation", "")
        state["task_raw"] = components.get("task", "")
        state["action_raw"] = components.get("action", "")
        state["result_raw"] = components.get("result", "")
        state["current_star_text"] = saved_text
        state["user_edited"] = True
        
        # CRITICAL FIX: If there was a pending question, mark it as answered
        if section_being_answered and had_pending_question:
            state["user_just_answered_for_section"] = section_being_answered
            state["last_question_asked"] = state.get("pending_question", "")
            state["last_answer_given"] = f"User edited and saved the {section_being_answered} section"
            state["last_section_improved"] = section_being_answered
            state["pending_question"] = None
            state["skip_generate"] = False  # Don't skip - we need to integrate the changes
            
            if VERBOSE:
                print(f"   ✅ Marked {section_being_answered} as answered via save")
        else:
            # No pending question - just a general edit
            state["skip_generate"] = True  # Skip generation for general edits
            
            if VERBOSE:
                print(f"   ✅ General edit (no pending question)")
        
        cl.user_session.set("state", state)
        
        if VERBOSE:
            print(f"   State updated with user edits")
            print(f"   situation_raw: {len(state['situation_raw'])} chars")
            print(f"   task_raw: {len(state['task_raw'])} chars")
            print(f"   action_raw: {len(state['action_raw'])} chars")
            print(f"   result_raw: {len(state['result_raw'])} chars")
        
        # Send confirmation
        await cl.Message(content="✅ Modifications sauvegardées!").send()
        
        # Automatically continue the workflow with the edited text
        if VERBOSE:
            print("   🔄 Continuing workflow with user-edited text...")
        
        result = await asyncio.to_thread(graph.invoke, state)
        cl.user_session.set("state", result)
        
        elem = cl.user_session.get("current_element")
        if result.get("current_star_text"):
            elem = await update_editable_text(result["current_star_text"])
        
        # Check if workflow completed
        if result.get("current_step") == "complete" or result.get("is_satisfactory"):
            final_message = f"🎉 **Mission STAR terminée !**\n\nVoici votre texte finalisé :\n\n{result.get('current_star_text', '')}"
            if elem:
                await cl.Message(content=final_message, elements=[elem]).send()
            else:
                await cl.Message(content=final_message).send()
        elif result.get("pending_question"):
            questions_left = MAX_TOTAL_QUESTIONS - result.get("total_questions_asked", 0)
            
            # Only show questions remaining if VERBOSE
            if VERBOSE:
                question_msg = f"{result.get('pending_question')}\n\n💡 ({questions_left} questions restantes)"
            else:
                question_msg = result.get('pending_question')
            
            if elem:
                await cl.Message(content=question_msg, elements=[elem]).send()
            else:
                await cl.Message(content=question_msg).send()
    else:
        await cl.Message(content="✅ Modifications sauvegardées!").send()


@cl.password_auth_callback
def auth_callback(username: str, password: str) -> cl.User | None:
    expected_username = os.getenv("APP_USERNAME", "")
    expected_password = os.getenv("APP_PASSWORD", "")
    
    if not expected_username or not expected_password:
        return cl.User(identifier="anonymous", metadata={"role": "user"})
    
    if username == expected_username and password == expected_password:
        return cl.User(identifier=username, metadata={"role": "user"})
    
    return None


@cl.on_chat_start
async def start():
    session_id = f"session-{uuid.uuid4()}"
    
    initial_state: STARState = {
        "job_description": "",
        "situation_raw": "",
        "task_raw": "",
        "action_raw": "",
        "result_raw": "",
        "current_star_text": "",
        "last_question_asked": None,
        "last_answer_given": None,
        "last_section_improved": None,
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
        "_recently_answered_section": None,
        "_consecutive_same_section_count": 0,
        "_last_improved_section": None,
    }

    cl.user_session.set("user_id", user_id)
    cl.user_session.set("session_id", session_id)
    cl.user_session.set("state", initial_state)
    cl.user_session.set("current_element", None)
    cl.user_session.set("extraction_done", False)
    
    welcome = prompts.WELCOME_MESSAGE
    await cl.Message(content=welcome).send()


@cl.on_message
async def main(message: cl.Message):
    if message.content.startswith("SAVE_STAR_TEXT:"):
        saved_text = message.content.replace("SAVE_STAR_TEXT:", "", 1)
        if 'star_text_saved' in _event_handlers:
            await _event_handlers['star_text_saved'](saved_text)
        return
    
    await handle_workflow_message(message)


async def handle_workflow_message(message: cl.Message):
    """Handle user messages through workflow."""
    
    state = cl.user_session.get("state")
    extraction_done = cl.user_session.get("extraction_done", False)
    
    # First message: extraction
    if not extraction_done:
        state["job_description"] = message.content
        
        if VERBOSE:
            print(f"\n📨 EXTRACTION INPUT LENGTH: {len(message.content)} chars")
            print(f"   Preview: {message.content[:200]}...")
        
        extracted = await asyncio.to_thread(extract_star_from_text, message.content)
        
        if VERBOSE:
            print(f"\n📦 EXTRACTED DATA:")
            for key, value in extracted.items():
                print(f"   {key}: {len(value) if value else 0} chars - {value[:80] if value else '(empty)'}...")
        
        # Check if extraction failed (all fields empty or very short)
        total_extracted = sum(len(v) for v in extracted.values())
        has_meaningful_content = any(len(v) > 50 for v in extracted.values())
        
        if total_extracted < 100 or not has_meaningful_content:
            # Extraction failed - input too short or not relevant
            if VERBOSE:
                print(f"\n⚠️ EXTRACTION FAILED - total extracted: {total_extracted} chars")
                print(f"   has_meaningful_content: {has_meaningful_content}")
            
            error_message = """❌ **Extraction impossible**

Votre message semble trop court ou ne contient pas assez d'informations pour créer une description STAR.

📝 **Veuillez insérer une description d'une expérience professionnelle, mission, ou une réalisation que vous souhaitez transformer au format STAR.**

**Exemple de contenu attendu :**
- Le contexte de votre mission (Situation)
- Vos responsabilités (Tâches)
- Les actions concrètes menées (Actions)
- Les résultats obtenus (Résultats)

Vous pouvez fournir un texte libre, formaté ou non. Merci de réessayer avec plus de détails ! 🙏"""
            
            await cl.Message(content=error_message).send()
            # Don't set extraction_done so user can try again
            return
        
        # Store in raw fields
        state["situation_raw"] = extracted.get("situation", "")
        state["task_raw"] = extracted.get("task", "")
        state["action_raw"] = extracted.get("action", "")
        state["result_raw"] = extracted.get("result", "")
        
        if VERBOSE:
            print(f"\n💾 STORED IN STATE:")
            print(f"   situation_raw: {len(state['situation_raw'])} chars")
            print(f"   task_raw: {len(state['task_raw'])} chars")
            print(f"   action_raw: {len(state['action_raw'])} chars")
            print(f"   result_raw: {len(state['result_raw'])} chars")
        
        # Generate initial text
        if any([state["situation_raw"], state["task_raw"], state["action_raw"], state["result_raw"]]):
            state["current_star_text"] = format_star_text_from_state(state)
            
            if VERBOSE:
                print(f"\n📄 FORMATTED TEXT:")
                print(state["current_star_text"])
        
        cl.user_session.set("extraction_done", True)
        cl.user_session.set("state", state)
        
        elem = None
        if state.get("current_star_text"):
            elem = await update_editable_text(state["current_star_text"])
        
        summary = "✅ **Extraction terminée !**\n\n💬 Tapez 'ok' pour commencer ou 'terminer' si le texte vous convient."
        
        if elem:
            await cl.Message(content=summary, elements=[elem]).send()
        else:
            await cl.Message(content=summary).send()
        
        return
    
    # Check for exit
    user_message_lower = message.content.lower().strip()
    exit_keywords = ["terminer", "termine", "exit", "quit", "done"]
    
    if any(keyword in user_message_lower for keyword in exit_keywords):
        state["user_wants_to_exit"] = True
        state["is_satisfactory"] = True
        state["current_step"] = "complete"
        cl.user_session.set("state", state)
        
        final_message = f"✅ **Mission STAR terminée !**\n\nVoici votre texte finalisé :\n\n{state.get('current_star_text', '')}"
        await cl.Message(content=final_message).send()
        return
    
    # Check for confirmation
    confirmation_keywords = ["ok", "oui", "yes", "continue", "go"]
    
    if any(keyword in user_message_lower for keyword in confirmation_keywords):
        result = await asyncio.to_thread(graph.invoke, state)
        cl.user_session.set("state", result)
        
        elem = cl.user_session.get("current_element")
        if result.get("current_star_text"):
            elem = await update_editable_text(result["current_star_text"])
        
        # Check if workflow completed immediately (already satisfactory)
        if result.get("current_step") == "complete" or result.get("is_satisfactory"):
            final_message = f"🎉 **Mission STAR terminée !**\n\nVoici votre texte finalisé :\n\n{result.get('current_star_text', '')}"
            if elem:
                await cl.Message(content=final_message, elements=[elem]).send()
            else:
                await cl.Message(content=final_message).send()
            return
        
        if result.get("pending_question"):
            questions_left = MAX_TOTAL_QUESTIONS - result.get("total_questions_asked", 0)
            
            # Only show questions remaining if VERBOSE
            if VERBOSE:
                question_msg = f"{result.get('pending_question')}\n\n💡 ({questions_left} questions restantes)"
            else:
                question_msg = result.get('pending_question')
            
            if elem:
                await cl.Message(content=question_msg, elements=[elem]).send()
            else:
                await cl.Message(content=question_msg).send()
        
        return
    
    # Normal answer processing
    section_to_improve = state.get("section_to_improve")
    
    if section_to_improve:
        # NEW: Store context for generation
        state["last_question_asked"] = state.get("pending_question", "")
        state["last_answer_given"] = message.content
        state["last_section_improved"] = section_to_improve
        
        # Update the raw field
        field_map = {
            "situation": "situation_raw",
            "task": "task_raw",
            "action": "action_raw",
            "result": "result_raw"
        }
        
        if section_to_improve in field_map:
            raw_field = field_map[section_to_improve]
            current = state.get(raw_field, "").strip()
            # Append with newline for context
            state[raw_field] = f"{current}\n{message.content}" if current else message.content
        
        state["user_just_answered_for_section"] = section_to_improve
        state["pending_question"] = None
        
        cl.user_session.set("state", state)
    
    # Invoke graph
    result = await asyncio.to_thread(graph.invoke, state)
    cl.user_session.set("state", result)
    
    elem = None
    if result.get("current_star_text"):
        elem = await update_editable_text(result["current_star_text"])
    
    if result.get("current_step") == "complete":
        final_message = f"🎉 **Mission STAR terminée !**\n\nVoici votre texte finalisé :\n\n{result.get('current_star_text', '')}"
        if elem:
            await cl.Message(content=final_message, elements=[elem]).send()
        else:
            await cl.Message(content=final_message).send()
    elif result.get("pending_question"):
        questions_left = MAX_TOTAL_QUESTIONS - result.get("total_questions_asked", 0)
        
        # Only show questions remaining if VERBOSE
        if VERBOSE:
            question_msg = f"{result.get('pending_question')}\n\n💡 ({questions_left} questions restantes)"
        else:
            question_msg = result.get('pending_question')
        
        if elem:
            await cl.Message(content=question_msg, elements=[elem]).send()
        else:
            await cl.Message(content=question_msg).send()


if __name__ == "__main__":
    pass