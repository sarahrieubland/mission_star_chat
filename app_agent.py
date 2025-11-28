"""
STAR Text Generator - Agentic Workflow with LangGraph and Chainlit

This application helps users create and iteratively improve STAR 
(Situation, Task, Action, Result) descriptions for job interviews.
"""

import asyncio
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
        print(f"   situation: {bool(state.get('situation'))}")
        print(f"   task: {bool(state.get('task'))}")
        print(f"   action: {bool(state.get('action'))}")
        print(f"   result: {bool(state.get('result'))}")
    
    # Determine which component needs more info
    if not state.get("situation"):
        prompt = prompts.SITUATION_PROMPT.format(input=state['job_description'])        
        question = call_llm(prompt)
        return {**state, "pending_question": question, "current_step": "gather_info"}
    
    elif not state.get("task"):
        prompt = prompts.TASK_PROMPT.format(input=state['situation'])
        question = call_llm(prompt)
        return {**state, "pending_question": question, "current_step": "gather_info"}
    
    elif not state.get("action"):
        prompt = prompts.ACTION_PROMPT.format(situation=state['situation'], task=state['task'])        
        question = call_llm(prompt)
        return {**state, "pending_question": question, "current_step": "gather_info"}
    
    elif not state.get("result"):
        prompt = prompts.RESULT_PROMPT.format(situation=state['situation'], task=state['task'], action=state['action'])        
        question = call_llm(prompt)
        return {**state, "pending_question": question, "current_step": "gather_info"}
    
    # All components gathered, move to generate
    return {**state, "current_step": "generate", "pending_question": None}


def generate_node(state: STARState) -> STARState:
    """Node that generates the STAR text based on available components"""
    
    if VERBOSE:
        print("\n🔄 GENERATE_NODE")
        print("-"*50)
        print("📋 CURRENT STATE:")
        print(f"   job_description: '{state.get('job_description', '')}'")
        print(f"   situation: '{state.get('situation', '')}'")
        print(f"   task: '{state.get('task', '')}'")
        print(f"   action: '{state.get('action', '')}'")
        print(f"   result: '{state.get('result', '')}'")
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
        print(f"   Components empty: {len(components) == 0}")
        for i, comp in enumerate(components):
            word_count = len(comp.split())
            print(f"   Component {i+1} word count: {word_count}")
        print("-"*50)
    
    prompt = prompts.GENERATE_STAR_PROMPT.format(input=components_text)
    
    if VERBOSE:
        print("📤 FULL PROMPT BEING SENT:")
        print(prompt)
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
        "iteration_count": state.get("iteration_count", 0) + 1
    }


def evaluate_node(state: STARState) -> STARState:
    """Node that autonomously evaluates the STAR text and decides next action"""
    
    if VERBOSE:
        print("\n🔄 EVALUATE_NODE")
    
    # Check if all components are gathered
    all_components = all([
        state.get("situation"), 
        state.get("task"), 
        state.get("action"), 
        state.get("result")
    ])
    
    if not all_components:
        # Still gathering info, continue to next component
        return {**state, "current_step": "gather_info", "is_satisfactory": False}
    
    # All components gathered - evaluate the final text
    prompt = prompts.EVALUATE_PROMPT.format(input=state['current_star_text'])
    response = call_llm(prompt)
    
    # Parse the response
    try:
        import json
        # Clean response if needed
        cleaned = response.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.split("```")[1]
            if cleaned.startswith("json"):
                cleaned = cleaned[4:]
        cleaned = cleaned.strip()
        
        evaluation = json.loads(cleaned)
        is_satisfactory = evaluation.get("is_satisfactory", False)
        question = evaluation.get("question")
        
        if is_satisfactory:
            return {**state, "is_satisfactory": True, "current_step": "complete", "pending_question": None}
        else:
            return {**state, "is_satisfactory": False, "current_step": "gather_info", "pending_question": question}
    
    except (json.JSONDecodeError, KeyError) as e:
        if VERBOSE:
            print(f"⚠️ Failed to parse evaluation response: {e}")
        # Default to satisfactory if parsing fails
        return {**state, "is_satisfactory": True, "current_step": "complete", "pending_question": None}


def complete_node(state: STARState) -> STARState:
    """Final node when the STAR text is complete"""
    if VERBOSE:
        print("\n✅ COMPLETE_NODE")
    return {**state, "current_step": "complete"}


# Router function to determine next step
def router(state: STARState) -> str:
    """Determine the next node based on current state"""
    
    step = state.get("current_step", "gather_info")
    
    if VERBOSE:
        print(f"\n🔀 ROUTER: current_step = {step}")
    
    if step == "complete":
        return "complete"
    
    if step == "gather_info":
        return END  # Stop and wait for user input
    
    if step == "generate":
        return "generate"
    
    if step == "evaluate":
        return "evaluate"
    
    return END


def after_gather_router(state: STARState) -> str:
    """Router after gather_info to decide if we generate or wait"""
    # Check if we just collected a new component (pending_question means we're asking)
    if state.get("pending_question"):
        return END  # Wait for user response
    else:
        return "generate"  # All info gathered, generate


def after_evaluate_router(state: STARState) -> str:
    """Router after evaluate to decide next step"""
    if state.get("is_satisfactory"):
        return "complete"
    elif state.get("pending_question"):
        return END  # Wait for user to answer the improvement question
    else:
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
    
    # After evaluating: complete, ask question, or continue gathering
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
    
    if VERBOSE:
        print("\n" + "="*50)
        print("📨 USER MESSAGE RECEIVED")
        print("="*50)
        print(f"   current_step: {current_step}")
        print(f"   user_input: {message.content[:100]}...")
    
    # Initial job description
    if not state.get("job_description"):
        state["job_description"] = message.content
        
    # Gathering STAR components
    elif current_step == "gather_info":
        if not state.get("situation"):
            state["situation"] = message.content
        elif not state.get("task"):
            state["task"] = message.content
        elif not state.get("action"):
            state["action"] = message.content
        elif not state.get("result"):
            state["result"] = message.content
        else:
            # Agent asked a clarifying question - update the relevant component
            # Append to the most recent component or handle as additional info
            if state.get("pending_question"):
                # Add the clarification to the result (or could be smarter about which component)
                state["result"] = state.get("result", "") + " " + message.content
    
    if VERBOSE:
        print("\n⚙️ Invoking graph...")
    
    # Run synchronous graph.invoke in a thread pool to avoid blocking
    result = await asyncio.to_thread(graph.invoke, state)
    
    if VERBOSE:
        print(f"✅ Graph invocation complete. New step: {result.get('current_step')}")
    
    # Update session state
    cl.user_session.set("state", result)
    
    # Send response based on result
    if result.get("current_step") == "complete":
        final_message = f"""🎉 **Votre mission STAR est terminée !**
Voici votre mission STAR finale (après {result.get('iteration_count', 1)} itération(s)) :
---
{result.get('current_star_text', '')}
 🌟"""
        
        await cl.Message(content=final_message).send()
    
    elif result.get("pending_question"):
        # Show current progress if we have generated text
        if result.get("current_star_text"):
            progress_msg = f"""📝 **Voici votre mission STAR en cours :**
---
{result.get('current_star_text', '')}
---
{result.get('pending_question')}"""

            await cl.Message(content=progress_msg).send()
        else:
            await cl.Message(content=result["pending_question"]).send()


if __name__ == "__main__":
    # This file is run by chainlit, not directly
    pass