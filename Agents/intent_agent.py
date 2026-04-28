# Agents/intent_agent.py
import os
from typing import Dict, Any
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, Field

INTENTS = [
    "check_schedule",
    "book_room",
    "control_hvac",
    "control_lights",
    "navigation",
    "energy_usage",
    "unknown"
]

class IntentResult(BaseModel):
    intent:     str   = Field(description="One label from the allowed intent list.")
    confidence: float = Field(description="Confidence from 0.0 to 1.0.")

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    api_key=os.getenv("GOOGLE_API_KEY")
)
structured_llm = llm.with_structured_output(IntentResult)

import re

def intent_agent_node(state: Dict[str, Any]) -> Dict[str, Any]:
    text  = state.get("cleaned_text", "").strip()
    lower = text.lower()

    if not text:
        return {"intent_data": {"category": "unknown", "confidence": 0.0}}

    # ✅ book_room FIRST — قبل check_schedule عشان "book" و"room" ميتاخدوش غلط
    book_keywords = ["book", "reserve", "ahgz", "7gz", "hagez", "booking",
                     "i need a room", "room for", "احجز", "حجز", "want to book",
                     "i want to book", "extra lecture", "extra session"]
    if any(kw in lower for kw in book_keywords):
        print(f"⚡ [Intent] Fast match → book_room")
        return {"intent_data": {"category": "book_room", "confidence": 0.97}}

    # ✅ Booking continuation — room code + date/time pattern
    # بيكشف رسائل زي "D210, 26/4/2026, from 2pm to 4pm"
    has_room_code = bool(re.search(r'\b[A-Za-z]\d{2,4}\b', text))          # D210, B101, etc.
    has_date      = bool(re.search(r'\b\d{1,2}[/\-\.]\d{1,2}[/\-\.]\d{2,4}\b', text))  # 26/4/2026
    has_time      = bool(re.search(r'\b\d{1,2}:\d{2}|\b\d{1,2}\s*(am|pm)\b', lower))   # 2pm, 14:00

    # لو فيه room code + (date أو time) → continuation لـ booking
    if has_room_code and (has_date or has_time):
        print(f"⚡ [Intent] Booking continuation detected → book_room")
        return {"intent_data": {"category": "book_room", "confidence": 0.95}}

    # لو فيه date + time بس (من غير room) وفي history سؤال عن booking
    session_history = state.get("session_history", [])
    if (has_date or has_time) and session_history:
        last_msgs = " ".join(
            m.get("content", "") for m in session_history[-4:]
            if m.get("role") in ("user", "assistant")
        ).lower()
        if any(kw in last_msgs for kw in ["book", "room", "lecture", "session", "reserve"]):
            print(f"⚡ [Intent] Date/time continuation in booking context → book_room")
            return {"intent_data": {"category": "book_room", "confidence": 0.92}}

    # check_schedule — بدون "room" عشان ميتعارضش مع book_room
    schedule_keywords = ["schedule", "lecture", "session", "class", "timetable",
                         "when", "what time", "today", "tomorrow", "monday", "tuesday",
                         "wednesday", "thursday", "friday", "saturday", "sunday",
                         "gadwal", "mo7adra", "ma3ad", "7ssa", "dr", "doctor",
                         "my lectures", "my classes", "جدول", "محاضرة"]
    if any(kw in lower for kw in schedule_keywords):
        print(f"⚡ [Intent] Fast match → check_schedule")
        return {"intent_data": {"category": "check_schedule", "confidence": 0.95}}

    keyword_map = {
        "control_hvac":   ["ac", "air", "temperature", "cool", "heat", "hvac", "cold", "hot"],
        "control_lights": ["light", "lamp", "bright", "dark", "nour", "daw"],
        "navigation":     ["where", "how to get", "location", "find", "feen", "fen"],
        "energy_usage":   ["energy", "electricity", "power", "consumption", "usage"],
    }

    for intent, keywords in keyword_map.items():
        if any(kw in lower for kw in keywords):
            print(f"⚡ [Intent] Fast match → {intent}")
            return {"intent_data": {"category": intent, "confidence": 0.95}}

    # LLM fallback
    try:
        sys_prompt = (
            f"You are an intent classifier for a Smart Building Management System.\n"
            f"Classify the user request into EXACTLY one of these intents:\n"
            f"{INTENTS}\n"
            f"IMPORTANT: If the user wants to BOOK or RESERVE a room → use 'book_room'.\n"
            f"If the user wants to CHECK their schedule → use 'check_schedule'.\n"
            f"Be strict — if not related to a building, use 'unknown'."
        )
        result = structured_llm.invoke([
            ("system", sys_prompt),
            ("human", text)
        ])
        intent = result.intent if result.intent in INTENTS else "unknown"
        print(f"🤖 [Intent] LLM → {intent} ({result.confidence:.2f})")
        return {"intent_data": {"category": intent, "confidence": result.confidence}}

    except Exception as e:
        print(f"❌ [Intent Error] {e}")
        return {"intent_data": {"category": "check_schedule", "confidence": 0.5}}