# Agents/interaction_agent.py
import logging
import os
import re
import requests
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Literal, Optional, Union

from pydantic import BaseModel, ConfigDict, Field, field_validator, ValidationError
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

logger = logging.getLogger(__name__)
_WHITESPACE_RE = re.compile(r"\s+")

# ─────────────────────────────────────────────
# Schemas
# ─────────────────────────────────────────────

class DisplayType(str, Enum):
    TEXT         = "text"
    NOTIFICATION = "notification"
    QUESTION     = "question"
    CONFIRMATION = "confirmation"
    ERROR        = "error"

class VoiceInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    type:      Literal["voice_input"]
    text:      str = Field(min_length=1)
    user_id:   str = Field(min_length=1)
    timestamp: Any

class AgentResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")
    type:    Literal["agent_response"]
    agent:   str = Field(min_length=1)
    message: str = Field(min_length=1)
    data:    Dict[str, Any] = Field(default_factory=dict)

class ClarificationRequired(BaseModel):
    model_config = ConfigDict(extra="forbid")
    type:     Literal["clarification_required"]
    agent:    str = Field(min_length=1)
    question: str = Field(min_length=1)

IncomingEvent = Union[VoiceInput, AgentResponse, ClarificationRequired]

class GuiOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    display_type: DisplayType
    title:        str
    message:      str
    data:         Dict[str, Any] = Field(default_factory=dict)

class ParsedEvent(BaseModel):
    event: IncomingEvent

    @field_validator("event", mode="before")
    @classmethod
    def parse_event(cls, value: Any) -> IncomingEvent:
        if not isinstance(value, dict):
            raise TypeError("incoming event must be a dict")
        event_type = value.get("type")
        if event_type == "voice_input":
            return VoiceInput.model_validate(value)
        if event_type == "agent_response":
            return AgentResponse.model_validate(value)
        if event_type == "clarification_required":
            return ClarificationRequired.model_validate(value)
        raise ValueError(f"Unsupported event type: {event_type}")

# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────

def _normalize_text(text: str) -> str:
    text = text.strip()
    text = _WHITESPACE_RE.sub(" ", text)
    text = text.replace(" ,", ",").replace(" .", ".")
    return text

def _get_weather() -> str:
    """الحرارة الحالية من Open-Meteo (Cairo)."""
    try:
        url  = "https://api.open-meteo.com/v1/forecast?latitude=30.06&longitude=31.25&current_weather=true"
        data = requests.get(url, timeout=5).json()
        temp = data["current_weather"]["temperature"]
        wcode = data["current_weather"].get("weathercode", -1)
        # ترجمة بسيطة للـ weather code
        conditions = {
            0: "Clear sky", 1: "Mainly clear", 2: "Partly cloudy", 3: "Overcast",
            45: "Foggy", 48: "Icy fog", 51: "Light drizzle", 61: "Light rain",
            71: "Light snow", 80: "Rain showers", 95: "Thunderstorm",
        }
        condition = conditions.get(wcode, "")
        return f"{temp}°C{f', {condition}' if condition else ''}"
    except Exception:
        return "unavailable"

# ─────────────────────────────────────────────
# LLM للـ direct_response_node
# ─────────────────────────────────────────────

_direct_llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.2,
    api_key=os.getenv("GOOGLE_API_KEY")
)

_DIRECT_SYSTEM = """\
You are Campus Agent, the interaction layer of a Smart Building Management System.
You handle simple, conversational queries that don't need a database lookup.

YOU CAN ANSWER:
- Current time and date
- Current weather / temperature
- HVAC and lights status queries (no actual control — just inform the user)
- Navigation inside the building (general guidance)
- Energy usage general questions
- Any small-talk or general building-related question

RULES:
- Be concise and friendly — 1-3 sentences max unless more detail is needed.
- Detect the user's language and reply in the SAME language (Arabic → Arabic, English → English).
- Never make up sensor data you don't have — use the provided context (time, weather).
- If the question is outside your scope, say so politely.
"""

# ─────────────────────────────────────────────
# Node 1 — user_interaction_node  (entry point)
# ─────────────────────────────────────────────

def user_interaction_node(state_input: Dict[str, Any]) -> Dict[str, Any]:
    raw_event = state_input.get("incoming_event", {})

    if not raw_event:
        return {**state_input, "route": "idle"}

    try:
        parsed = ParsedEvent(event=raw_event).event
    except (ValidationError, Exception) as e:
        gui = GuiOutput(
            display_type=DisplayType.ERROR,
            title="Input Error",
            message=f"Could not understand the input: {str(e)}"
        )
        return {
            **state_input,
            "gui_message":    gui.model_dump(),
            "incoming_event": {},
            "route":          "to_gui",
        }

    if isinstance(parsed, VoiceInput):
        clean = _normalize_text(parsed.text)
        return {
            **state_input,
            "cleaned_text":   clean,
            "incoming_event": {},
            "gui_message":    None,
            "route":          "to_intent",
        }

    elif isinstance(parsed, ClarificationRequired):
        gui = GuiOutput(
            display_type=DisplayType.QUESTION,
            title=f"Question from {parsed.agent}",
            message=parsed.question
        )
        return {
            **state_input,
            "pending_question": parsed.question,
            "gui_message":      gui.model_dump(),
            "incoming_event":   {},
            "route":            "await_user",
        }

    elif isinstance(parsed, AgentResponse):
        gui = GuiOutput(
            display_type=DisplayType.CONFIRMATION,
            title=f"Update from {parsed.agent}",
            message=parsed.message,
            data=parsed.data
        )
        return {
            **state_input,
            "gui_message":    gui.model_dump(),
            "incoming_event": {},
            "route":          "to_gui",
        }

    return {**state_input, "route": "idle"}


# ─────────────────────────────────────────────
# Node 2 — direct_response_node
# بيتم استدعاؤه من الـ planner للـ intents البسيطة:
#   control_hvac | control_lights | navigation | energy_usage | unknown
# ─────────────────────────────────────────────

def direct_response_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    يرد مباشرة على الأسئلة البسيطة بدون DB lookup.
    - الوقت والتاريخ   → hardcoded من datetime
    - الحرارة          → hardcoded من weather API
    - باقي الأسئلة     → Gemini مع context (وقت + حرارة + اسم المستخدم)
    """
    cleaned_text = state.get("cleaned_text", "").strip()
    user_name    = state.get("user_id", "User")
    intent       = state.get("intent_data", {}).get("category", "unknown")
    session_hist = state.get("session_history", [])  # لو موجود في الـ state

    if not cleaned_text:
        return {"reports": ["Response: Please type your question."]}

    now     = datetime.now()
    weather = _get_weather()
    lower   = cleaned_text.lower()

    # ── Hardcoded: وقت ──────────────────────────────────────────────
    time_keywords = ["time", "clock", "sa3a", "sa3et", "الساعة", "كام الساعة", "what time"]
    if any(kw in lower for kw in time_keywords) and intent not in ("check_schedule", "book_room"):
        reply = f"⏰ الساعة دلوقتي {now.strftime('%I:%M %p')} — {now.strftime('%A, %d %B %Y')}."
        print(f"⚡ [Direct] Hardcoded time response.")
        return {"reports": [f"Response: {reply}"]}

    # ── Hardcoded: حرارة / طقس ──────────────────────────────────────
    weather_keywords = [
        "weather", "temperature", "temp", "hot", "cold", "heat",
        "درجة", "حرارة", "الجو", "طقس", "برد", "حر"
    ]
    if any(kw in lower for kw in weather_keywords) and intent not in ("check_schedule", "book_room"):
        reply = f"🌡️ درجة الحرارة دلوقتي {weather}."
        print(f"⚡ [Direct] Hardcoded weather response.")
        return {"reports": [f"Response: {reply}"]}

    # ── Gemini: باقي الأسئلة (HVAC / lights / nav / energy / unknown) ──
    context_line = (
        f"Current context:\n"
        f"- User    : {user_name}\n"
        f"- Intent  : {intent}\n"
        f"- Time    : {now.strftime('%I:%M %p')}\n"
        f"- Date    : {now.strftime('%A, %d %B %Y')}\n"
        f"- Weather : {weather}"
    )

    messages = [SystemMessage(content=f"{_DIRECT_SYSTEM}\n\n{context_line}")]

    # نضيف الـ session history لو موجود (للـ multi-turn)
    for msg in session_hist:
        if msg["role"] == "user":
            messages.append(HumanMessage(content=msg["content"]))
        elif msg["role"] == "assistant":
            messages.append(AIMessage(content=msg["content"]))

    messages.append(HumanMessage(content=cleaned_text))

    try:
        response  = _direct_llm.invoke(messages)
        final_msg = response.content if isinstance(response.content, str) else str(response.content)
        print(f"🤖 [Direct] Gemini responded for intent: {intent}")
        return {"reports": [f"Response: {final_msg}"]}

    except Exception as e:
        print(f"❌ [Direct Response Error] {e}")
        return {"reports": ["Response: Sorry, something went wrong. Please try again."]}


# ─────────────────────────────────────────────
# Router
# ─────────────────────────────────────────────

def route_next_step(state: Dict[str, Any]) -> str:
    return state.get("route", "idle")