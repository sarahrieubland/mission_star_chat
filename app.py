"""
Mission STAR chat project

(Chainlit + LangChain + Langfuse + OpenAI)

Description:
- In-memory session store (lost on restart) as requested.
- Uses LangChain PromptTemplate + LLMChain for prompt management (easy to swap prompts later).
- Integrates Langfuse for prompt/version tracing (optional; app still runs if LANGFUSE not configured).
- Uses OpenAI ChatCompletion under the hood via langchain's OpenAI wrapper (Chat-style).
- Chainlit provides a minimal UI; the agent loops asking targeted follow-ups until STAR fields are satisfactory.
"""

import os
import re
import json
import uuid
import asyncio
from typing import Dict, Any, Optional

import chainlit as cl
from dotenv import load_dotenv

from config import prompts

# Modern LangChain imports
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

# Langfuse callback handler for LangChain
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

# --- Helpers: prompts & LLM setup ---
MODEL_NAME = os.getenv("POC_MODEL", "gpt-4")
TEMPERATURE = float(os.getenv("POC_TEMP", "0.0"))

# Modern ChatOpenAI wrapper
llm = ChatOpenAI(model=MODEL_NAME, temperature=TEMPERATURE)

# Attach Langfuse callback if available and configured
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

# Output parser
output_parser = StrOutputParser()

# Build prompt templates from the prompts module
EXTRACTION_PROMPT = PromptTemplate.from_template(prompts.EXTRACTION_PROMPT_TEMPLATE)
QUESTION_PROMPT = PromptTemplate.from_template(prompts.QUESTION_PROMPT_TEMPLATE)
UPDATE_PROMPT = PromptTemplate.from_template(prompts.UPDATE_PROMPT_TEMPLATE)

# Build chains using LCEL (LangChain Expression Language)
EXTRACTION_CHAIN = EXTRACTION_PROMPT | llm | output_parser
QUESTION_CHAIN = QUESTION_PROMPT | llm | output_parser
UPDATE_CHAIN = UPDATE_PROMPT | llm | output_parser

# --- Utility functions ---

def parse_json_from_text(text: str) -> Optional[Dict[str, Any]]:
    """Extract JSON object from LLM text response and parse it."""
    m = re.search(r"\{.*\}\s*$", text, re.DOTALL)
    if not m:
        # try to find first { ... }
        m = re.search(r"\{.*?\}", text, re.DOTALL)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except Exception:
        # try to be tolerant: replace single quotes, remove trailing commas
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


def identify_missing(star: Dict[str, Any]) -> Optional[str]:
    """Return the single most important missing/weak field, or None if all ok."""
    # fields in priority order
    priority = ["result", "action", "task", "situation"]
    for f in priority:
        entry = star.get(f, {})
        text = entry.get("text", "") if isinstance(entry, dict) else entry
        conf = entry.get("confidence", 0.0) if isinstance(entry, dict) else (0.0 if not text else 0.5)
        # treat as missing if empty or low confidence
        if not text or conf < 0.6:
            # for result, also ask if no numeric metric
            if f == "result" and text and has_result_metric(text):
                continue
            return f
    return None

# --- Core LLM wrappers ---

async def extract_star(input_text: str) -> Optional[Dict[str, Any]]:
    """Call the extraction chain and return parsed STAR JSON."""
    # Use ainvoke for async execution with LCEL
    res = await EXTRACTION_CHAIN.ainvoke(
        {"input_text": input_text},
        config={"callbacks": lc_callbacks}
    )
    parsed = parse_json_from_text(res)
    return parsed

async def generate_followup_question(input_text: str, star_json: Dict[str, Any], field: str) -> str:
    res = await QUESTION_CHAIN.ainvoke(
        {
            "input_text": input_text,
            "star_json": json.dumps(star_json),
            "field": field
        },
        config={"callbacks": lc_callbacks}
    )
    return res.strip()

async def update_star_field(field: str, star_json: Dict[str, Any], answer: str) -> Optional[Dict[str, Any]]:
    res = await UPDATE_CHAIN.ainvoke(
        {
            "field": field,
            "star_json": json.dumps(star_json),
            "answer": answer
        },
        config={"callbacks": lc_callbacks}
    )
    parsed = parse_json_from_text(res)
    return parsed

# --- Chainlit handlers ---

@cl.on_chat_start
async def on_chat_start():
    # create a new session id and store in chainlit user_session
    session_id = str(uuid.uuid4())
    cl.user_session.set("session_id", session_id)
    SESSIONS[session_id] = {
        "id": session_id,
        "raw_inputs": [],
        "star": None,
        "status": "new",
        "awaiting_field": None
    }
    await cl.Message(content=prompts.WELCOME_MESSAGE).send()


@cl.on_message
async def main(message: cl.Message):
    session_id = cl.user_session.get("session_id")
    if not session_id or session_id not in SESSIONS:
        await cl.Message(prompts.MSG_SESSION_NOT_FOUND).send()
        return

    session = SESSIONS[session_id]
    text = message.content.strip()

    # If we are waiting for a followup for a particular field
    if session.get("awaiting_field"):
        field = session["awaiting_field"]
        session["raw_inputs"].append({"role": "user", "content": text})
        await cl.Message(prompts.MSG_UPDATING_FIELD.format(field=field)).send()
        updated = await update_star_field(field, session["star"], text)
        if updated:
            session["star"] = updated
            session["awaiting_field"] = None
            # check next missing
            next_field = identify_missing(updated)
            if next_field:
                q = await generate_followup_question("\n".join([r["content"] for r in session["raw_inputs"] if isinstance(r, dict)]), updated, next_field)
                session["awaiting_field"] = next_field
                await cl.Message(content=prompts.RESPONSE_UPDATE_WITH_FOLLOWUP.format(
                    star_json=json.dumps(updated, indent=2),
                    question=q
                )).send()
            else:
                session["status"] = "complete"
                await cl.Message(content=prompts.RESPONSE_COMPLETE.format(
                    star_json=json.dumps(updated, indent=2)
                )).send()
        else:
            await cl.Message(prompts.MSG_UPDATE_PARSE_ERROR).send()
        return

    # Otherwise this is an initial job description or a new item
    session["raw_inputs"].append({"role": "user", "content": text})
    await cl.Message(prompts.MSG_CREATING_DRAFT).send()
    draft = await extract_star(text)
    if not draft:
        await cl.Message(prompts.MSG_PARSE_ERROR).send()
        return

    session["star"] = draft
    session["status"] = "draft"

    # Decide if we need follow-up
    missing = identify_missing(draft)
    if missing:
        q = await generate_followup_question(text, draft, missing)
        session["awaiting_field"] = missing
        await cl.Message(content=prompts.RESPONSE_WITH_FOLLOWUP.format(
            star_json=json.dumps(draft, indent=2),
            field=missing,
            question=q
        )).send()
    else:
        session["status"] = "complete"
        await cl.Message(content=prompts.RESPONSE_DRAFT_COMPLETE.format(
            star_json=json.dumps(draft, indent=2)
        )).send()

# --- Small helper CLI endpoint for quick testing (optional) ---
if __name__ == "__main__":
    print("This file is intended to be run with: chainlit run star_agent_poc_updated.py -w")