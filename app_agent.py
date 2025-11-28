"""
STAR Text Generator - Agentic Workflow with LangGraph and Chainlit

This application helps users create and iteratively improve STAR 
(Situation, Task, Action, Result) descriptions for job interviews.
"""

import asyncio
import json
import os
from typing import TypedDict, Literal, Optional

import chainlit as cl
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END

from config import prompts


# --- LLM setup ---
MODEL_NAME = os.getenv("POC_MODEL", "gpt-4o-mini")
TEMPERATURE = float(os.getenv("POC_TEMP", "0.0"))
VERBOSE = os.getenv("VERBOSE", "true").lower() == "true"

# Initialize LangChain OpenAI client
llm = ChatOpenAI(model=MODEL_NAME, temperature=TEMPERATURE)


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


def call_llm(prompt: str, system: str = prompts.AGENT_SYSTEM_PROMPT) -> str:
    """Make an LLM call with the given prompt"""
    messages = [
        ("system", system),
        ("user", prompt)
    ]
    
    if VERBOSE:
        print("\n" + "="*50)
        print("📤 LLM CALL")
        print("="*50)
        print(f"🔧 System: {system[:100]}..." if len(system) > 100 else f"🔧 System: {system}")
        print(f"📝 Prompt: {prompt}")
        print("-"*50)
    
    response = llm.invoke(messages)
    
    if VERBOSE:
        print(f"📥 Response: {response.content}")
        print("="*50 + "\n")
    
    return response.content


# Node functions for the LangGraph workflow

def gather_info_node(state: STARState) -> STARState:
    """Node that gathers information about a specific STAR component"""
    
    if VERBOSE:
        print("\n🔄 GATHER_INFO_NODE")
        print(f"   situation: '{state.get('situation', '')[:50]}...' ({bool(state.get('situation'))})")
        print(f"   task: '{state.get('task', '')[:50]}...' ({bool(state.get('task'))})")
        print(f"   action: '{state.get('action', '')[:50]}...' ({bool(state.get('action'))})")
        print(f"   result: '{state.get('result', '')[:50]}...' ({bool(state.get('result'))})")
        print(f"   section_to_improve: {state.get('section_to_improve')}")
    
    # If evaluate told us which section to improve, ask about that specific section
    section_to_improve = state.get("section_to_improve")
    
    if section_to_improve == "situation":
        prompt = prompts.SITUATION_PROMPT.format(input=state['job_description'])
        question = call_llm(prompt)
        if VERBOSE:
            print(f"   📌 Asking to improve SITUATION")
        return {**state, "pending_question": question, "section_to_improve": "situation"}
    
    elif section_to_improve == "task":
        prompt = prompts.TASK_PROMPT.format(input=state['situation'])
        question = call_llm(prompt)
        if VERBOSE:
            print(f"   📌 Asking to improve TASK")
        return {**state, "pending_question": question, "section_to_improve": "task"}
    
    elif section_to_improve == "action":
        prompt = prompts.ACTION_PROMPT.format(situation=state['situation'], task=state['task'])
        question = call_llm(prompt)
        if VERBOSE:
            print(f"   📌 Asking to improve ACTION")
        return {**state, "pending_question": question, "section_to_improve": "action"}
    
    elif section_to_improve == "result":
        prompt = prompts.RESULT_PROMPT.format(situation=state['situation'], task=state['task'], action=state['action'])
        question = call_llm(prompt)
        if VERBOSE:
            print(f"   📌 Asking to improve RESULT")
        return {**state, "pending_question": question, "section_to_improve": "result"}
    
    # Normal flow: determine which component needs info (first time through)
    if not state.get("situation"):
        prompt = prompts.SITUATION_PROMPT.format(input=state['job_description'])        
        question = call_llm(prompt)
        if VERBOSE:
            print(f"   📌 Asking for SITUATION (first time)")
        return {**state, "pending_question": question, "section_to_improve": "situation"}
    
    elif not state.get("task"):
        prompt = prompts.TASK_PROMPT.format(input=state['situation'])
        question = call_llm(prompt)
        if VERBOSE:
            print(f"   📌 Asking for TASK (first time)")
        return {**state, "pending_question": question, "section_to_improve": "task"}
    
    elif not state.get("action"):
        prompt = prompts.ACTION_PROMPT.format(situation=state['situation'], task=state['task'])        
        question = call_llm(prompt)
        if VERBOSE:
            print(f"   📌 Asking for ACTION (first time)")
        return {**state, "pending_question": question, "section_to_improve": "action"}
    
    elif not state.get("result"):
        prompt = prompts.RESULT_PROMPT.format(situation=state['situation'], task=state['task'], action=state['action'])        
        question = call_llm(prompt)
        if VERBOSE:
            print(f"   📌 Asking for RESULT (first time)")
        return {**state, "pending_question": question, "section_to_improve": "result"}
    
    # All components gathered, clear section_to_improve
    if VERBOSE:
        print(f"   ✅ All components gathered")
    return {**state, "pending_question": None, "section_to_improve": None}


def generate_node(state: STARState) -> STARState:
    """Node that generates the STAR text based on available components"""
    
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
    
    # Build the prompt based on which components are available
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
    
    if VERBOSE:
        print("📝 COMPONENTS TO GENERATE FROM:")
        print(components_text)
        print("-"*50)
        print(f"   Number of components: {len(components)}")
        for i, comp in enumerate(components):
            word_count = len(comp.split())
            print(f"   Component {i+1} word count: {word_count}")
        print("-"*50)
    
    prompt = prompts.GENERATE_STAR_PROMPT.format(input=components_text)
    
    if VERBOSE:
        print("📤 FULL PROMPT BEING SENT:")
        print(prompt[:200] + "..." if len(prompt) > 200 else prompt)
        print("-"*50)
    
    star_text = call_llm(prompt)
    
    if VERBOSE:
        print("📥 GENERATED STAR TEXT:")
        print(star_text)
        print("="*50 + "\n")
    
    return {
        **state, 
        "current_star_text": star_text, 
        "current_step": "evaluate",
        "iteration_count": state.get("iteration_count", 0) + 1,
        "section_to_improve": None  # Clear after generating
    }


def evaluate_node(state: STARState) -> STARState:
    """Node that autonomously evaluates the STAR text and decides which section needs improvement"""
    
    if VERBOSE:
        print("\n🔄 EVALUATE_NODE")
        print(f"   iteration_count: {state.get('iteration_count', 0)}")
    
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
    
    # All components gathered - evaluate the final text
    prompt = prompts.EVALUATE_PROMPT.format(input=state['current_star_text'])
    
    if VERBOSE:
        print("   📊 Evaluating STAR text...")
    
    response = call_llm(prompt)
    
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
        section_to_improve = evaluation.get("section_to_improve")  # "situation", "task", "action", "result", or None
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
                "section_to_improve": None
            }
        else:
            return {
                **state, 
                "is_satisfactory": False, 
                "current_step": "gather_info",
                "pending_question": question,
                "section_to_improve": section_to_improve
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
            "section_to_improve": None
        }


def complete_node(state: STARState) -> STARState:
    """Final node when the STAR text is complete"""
    if VERBOSE:
        print("\n✅ COMPLETE_NODE")
        print(f"   Final iteration count: {state.get('iteration_count', 0)}")
    return {**state, "current_step": "complete"}


# Router functions

def after_gather_router(state: STARState) -> str:
    """Router after gather_info - generate if we have at least one component"""
    has_any_component = any([
        state.get("situation"),
        state.get("task"),
        state.get("action"),
        state.get("result")
    ])
    
    if VERBOSE:
        print(f"\n🔀 AFTER_GATHER_ROUTER")
        print(f"   has_any_component: {has_any_component}")
        print(f"   pending_question: {bool(state.get('pending_question'))}")
    
    if state.get("pending_question"):
        # We have a question to ask, stop and wait for user
        if VERBOSE:
            print(f"   → END (waiting for user response)")
        return END
    elif has_any_component:
        if VERBOSE:
            print(f"   → generate")
        return "generate"
    else:
        if VERBOSE:
            print(f"   → END (no components yet)")
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
    
    # After gathering info: either wait for user or generate
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


# Chainlit handlers

@cl.on_chat_start
async def start():
    """Initialize the chat session"""
    
    # Initialize state
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
        "is_satisfactory": False
    }
    
    # Store state in session
    cl.user_session.set("state", initial_state)
    
    # Welcome message
    welcome = prompts.WELCOME_MESSAGE
    await cl.Message(content=welcome).send()


@cl.on_message
async def main(message: cl.Message):
    """Handle incoming messages"""
    
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
    result = await asyncio.to_thread(graph.invoke, state)
    
    if VERBOSE:
        print(f"✅ Graph invocation complete.")
        print(f"   New step: {result.get('current_step')}")
        print(f"   section_to_improve: {result.get('section_to_improve')}")
        print(f"   is_satisfactory: {result.get('is_satisfactory')}")
    
    # Update session state
    cl.user_session.set("state", result)
    
    # Send response based on result
    if result.get("current_step") == "complete":
        final_message = f"""🎉 **Votre mission STAR est terminée !**

Voici votre mission STAR finale (après {result.get('iteration_count', 1)} itération(s)) :

---

{result.get('current_star_text', '')}

---

🌟"""
        
        await cl.Message(content=final_message).send()
    
    elif result.get("pending_question"):
        # Show current progress if we have generated text
        if result.get("current_star_text"):
            section_name = result.get("section_to_improve", "").upper() if result.get("section_to_improve") else ""
            progress_msg = f"""📝 **Voici votre mission STAR en cours :**

---

{result.get('current_star_text', '')}

---

{"🔍 **Amélioration de la section " + section_name + ":**" if section_name else ""}

{result.get('pending_question')}"""

            await cl.Message(content=progress_msg).send()
        else:
            await cl.Message(content=result["pending_question"]).send()


if __name__ == "__main__":
    # This file is run by chainlit, not directly
    pass