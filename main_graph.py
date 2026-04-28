import operator
import os
import json
import traceback
from datetime import datetime, timezone
from typing import List, Annotated, TypedDict, Dict, Any
from dotenv import load_dotenv
from Agents.context_agent import context_agent_node

load_dotenv(override=True)

from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

from DBs.azure_sql   import verify_user_identity, get_lecturer_full_context, get_student_full_context, resolve_user, resolve_user_name
from DBs.azure_cosmos import log_to_cosmos

from Agents.interaction_agent import (
    user_interaction_node,
    direct_response_node,
    route_next_step,
)
from Agents.intent_agent     import intent_agent_node
from Agents.scheduling_agent import scheduling_agent, SYSTEM_PROMPT, _extract_text

SESSION_HISTORY: list = []
USER_INFO: dict = {}  # stores full user info after login

# FIX 1: Cache the context once per session instead of fetching on every message
CACHED_CONTEXT: list = []
CACHED_USER: str = ""


class BMSGraphState(TypedDict):
    conversation_id:  str
    user_id:          str
    history:          Annotated[list, operator.add]
    incoming_event:   Dict[str, Any]
    cleaned_text:     str
    route:            str
    intent_data:      Dict[str, Any]
    context_data:     List[Dict[str, Any]]
    reports:          Annotated[List[str], operator.add]
    iteration_count:  int
    gui_message:      Dict[str, Any]
    pending_question: str


# FIX 3: Compress context to a compact summary instead of sending raw JSON
def compress_context(context_data: list) -> str:
    """
    Convert API schedule response to a compact summary to reduce token usage.
    API fields: day, time, courseName, roomName, type, instructorName, id
    """
    if not context_data:
        return "No schedule data available."

    lines = []
    for session in context_data:
        day       = session.get("day", "?")
        time      = session.get("time", "?")
        course    = session.get("courseName", "?")
        course_id = session.get("courseId", "")
        room      = session.get("roomName", "?")
        stype     = session.get("type", "")
        sid       = session.get("id", "")

        type_str      = f" [{stype}]" if stype else ""
        id_str        = f" (ID:{sid})" if sid else ""
        course_id_str = f" [courseId:{course_id}]" if course_id else ""
        lines.append(f"- {day} {time}: {course}{course_id_str}{type_str} @ {room}{id_str}")

    return "\n".join(lines)


def intent_analyzer_node(state: BMSGraphState):
    state_with_history = {**state, "session_history": SESSION_HISTORY}
    result   = intent_agent_node(state_with_history)
    category = result.get("intent_data", {}).get("category", "unknown")
    conf     = result.get("intent_data", {}).get("confidence", 0.0)
    print(f"🎯 [Intent] → {category}  (conf: {conf:.2f})")
    return result


def context_fetcher_node(state: BMSGraphState):
    """Return cached context — populated once at login for speed."""
    global CACHED_CONTEXT
    if CACHED_CONTEXT:
        print(f"⚡ [Context] Using cached context ({len(CACHED_CONTEXT)} sessions) — no DB call.")
        return {
            "context_data": CACHED_CONTEXT,
            "reports":      [f"[Context] Loaded {len(CACHED_CONTEXT)} sessions from cache."],
        }

    # Fallback cache miss
    user_name = state.get("user_id", "Unknown")
    role      = USER_INFO.get("role", "Instructor")
    print(f"🕵️  [Context] Cache miss — fetching for {user_name} ({role})...")
    try:
        if role.lower() == "student":
            db_data = get_student_full_context(USER_INFO.get("id", 0))
        else:
            db_data = get_lecturer_full_context(user_name)

        if db_data and "error" not in db_data[0]:
            CACHED_CONTEXT = db_data
            print(f"✅ [Context] Found {len(db_data)} sessions. Cached.")
            return {"context_data": db_data, "reports": [f"[Context] Loaded {len(db_data)} sessions."]}
        print("⚠️  [Context] No sessions found.")
        return {"context_data": [], "reports": ["[Context] No sessions found."]}
    except Exception as e:
        print(f"❌ [Context Error] {e}")
        return {"context_data": [], "reports": [f"[Context Error] {str(e)}"]}


def planner_node(state: BMSGraphState):
    count = state.get("iteration_count", 0) + 1
    if count > 10:
        raise RuntimeError("Infinite loop guard triggered.")
    category = state.get("intent_data", {}).get("category", "unknown")
    conf     = state.get("intent_data", {}).get("confidence", 0.0)
    print(f"🧠 [Planner] Routing → '{category}'  (conf: {conf:.2f})")
    return {"iteration_count": count}


def call_scheduling_agent(state: BMSGraphState):
    global SESSION_HISTORY

    user_name    = state.get("user_id", "User")
    context_data = state.get("context_data", [])
    cleaned_text = state.get("cleaned_text", "").strip()

    if not cleaned_text:
        return {"reports": ["Response: Please type your question."]}

    # FIX 3: Use compressed context summary instead of raw JSON dump
    context_str = compress_context(context_data)

    now = datetime.now()

    role = USER_INFO.get("role", "Instructor")
    awareness = (
        f"{SYSTEM_PROMPT}\n\n"
        f"CURRENT SESSION:\n"
        f"- User      : {user_name}\n"
        f"- Role      : {role}\n"
        f"- Time      : {now.strftime('%I:%M %p')}\n"
        f"- Date      : {now.strftime('%A, %d %B %Y')}\n\n"
        f"SCHEDULE FOR {user_name.upper()}:\n{context_str}"
    )

    # Gemini requires conversation to start with HumanMessage, not SystemMessage.
    # Awareness (system prompt + context) is always embedded in the FIRST HumanMessage.
    # Current user message is ALWAYS present — even when history is empty.

    # FIX 2: Only send last 6 history messages (3 exchanges) to reduce token load
    recent_history = SESSION_HISTORY[-6:] if len(SESSION_HISTORY) > 6 else SESSION_HISTORY

    # Current user message with full system context embedded
    current_msg = HumanMessage(content=f"{awareness}\n\n---\n\nUser question: {cleaned_text}")

    if recent_history:
        # Build history messages first, append current at the end
        messages = []
        for msg in recent_history:
            if msg["role"] == "user":
                messages.append(HumanMessage(content=msg["content"]))
            elif msg["role"] == "assistant":
                messages.append(AIMessage(content=msg["content"]))
        messages.append(current_msg)
    else:
        # No history yet (first message in session) → just the current message
        messages = [current_msg]

    print(f"📜 [History] Sending {len(recent_history)} messages from history + current input")

    try:
        result   = scheduling_agent.invoke({"messages": messages})
        last_msg = result["messages"][-1]
        final_msg = _extract_text(last_msg.content)

        SESSION_HISTORY.append({"role": "user",      "content": cleaned_text})
        SESSION_HISTORY.append({"role": "assistant", "content": final_msg})

        # FIX 2: Keep only last 6 messages (trim after appending)
        if len(SESSION_HISTORY) > 6:
            SESSION_HISTORY = SESSION_HISTORY[-6:]

        return {"reports": [f"Response: {final_msg}"]}

    except Exception as e:
        traceback.print_exc()
        print(f"❌ [Scheduling Agent Error] {e}")
        return {"reports": ["Response: Sorry, something went wrong. Please try again."]}


def final_logger_node(state: BMSGraphState):
    reports = state.get("reports", [])
    if reports:
        try:
            log_to_cosmos(state.get("cleaned_text", "N/A"), reports)
        except Exception as e:
            print(f"⚠️  [Logger] {e}")
    return state


SCHEDULING_INTENTS = {"check_schedule", "book_room"}


def route_to_agent(state: BMSGraphState) -> str:
    category = state.get("intent_data", {}).get("category", "unknown")
    if category in SCHEDULING_INTENTS:
        return "scheduling_agent"
    return "direct_response"


workflow = StateGraph(BMSGraphState)
workflow.add_node("interaction",      user_interaction_node)
workflow.add_node("intent_analyzer",  intent_analyzer_node)
workflow.add_node("context_fetcher",  context_fetcher_node)
workflow.add_node("planner",          planner_node)
workflow.add_node("scheduling_agent", call_scheduling_agent)
workflow.add_node("direct_response",  direct_response_node)
workflow.add_node("logger",           final_logger_node)

workflow.set_entry_point("interaction")

workflow.add_conditional_edges(
    "interaction",
    route_next_step,
    {
        "to_intent":  "intent_analyzer",
        "to_gui":     "logger",
        "await_user": END,
        "idle":       END,
    }
)

workflow.add_edge("intent_analyzer", "context_fetcher")
workflow.add_edge("context_fetcher", "planner")

workflow.add_conditional_edges(
    "planner",
    route_to_agent,
    {
        "scheduling_agent": "scheduling_agent",
        "direct_response":  "direct_response",
    }
)

workflow.add_edge("scheduling_agent", "logger")
workflow.add_edge("direct_response",  "logger")
workflow.add_edge("logger", END)

orchestrator = workflow.compile()


if __name__ == "__main__":
    print("\n" + "="*60)
    print("🎓 CAMPUS AGENT — SMART BUILDING SYSTEM (ONLINE)")
    print("="*60)

    current_user = None
    while not current_user:
        attempt = input("\n🔐 Authentication Required. Enter Name/ID: ").strip()
        if verify_user_identity(attempt):
            user_info    = resolve_user(attempt)
            current_user = resolve_user_name(attempt)   # ✅ دايما بيجيب الاسم الحقيقي من الـ DB
            role         = user_info.get("role", "Instructor")
            USER_INFO.update(user_info)
            print(f"✅ Access Granted. Welcome, {current_user}! ({role})")

            # Load and cache context once at login
            print("⏳ Loading your schedule...")
            try:
                if role.lower() == "student":
                    db_data = get_student_full_context(user_info.get("id", 0))
                else:
                    db_data = get_lecturer_full_context(current_user)

                if db_data and "error" not in db_data[0]:
                    CACHED_CONTEXT.clear()
                    CACHED_CONTEXT.extend(db_data)
                    print(f"✅ Schedule loaded: {len(CACHED_CONTEXT)} sessions cached.")
                else:
                    print("⚠️  No schedule data found.")
            except Exception as e:
                print(f"⚠️  Could not load schedule: {e}")
        else:
            print("❌ Access Denied: User not found in system.")

    SESSION_HISTORY.clear()

    while True:
        query = input(f"\n👤 {current_user}: ").strip()
        if not query or query.lower() in ["exit", "quit"]:
            print("👋 Goodbye!")
            break

        initial_state: BMSGraphState = {
            "conversation_id":  f"BMS_{datetime.now().strftime('%H%M%S')}",
            "user_id":          current_user,
            "incoming_event": {
                "type":      "voice_input",
                "text":      query,
                "user_id":   current_user,
                "timestamp": datetime.now(timezone.utc).isoformat()
            },
            "reports":          [],
            "iteration_count":  0,
            "history":          [],
            "context_data":     [],
            "intent_data":      {},
            "cleaned_text":     "",
            "route":            "",
            "gui_message":      None,
            "pending_question": "",
        }

        print("\n⚙️  Processing", end="", flush=True)

        final_ans = ""
        for chunk in orchestrator.stream(initial_state, subgraphs=False):
            for node, values in chunk.items():
                print(f" ➔ {node}", end="", flush=True)

                if "reports" in values and values["reports"]:
                    last = values["reports"][-1]
                    if "Response:" in last:
                        final_ans = last.split("Response:", 1)[-1].strip()

                if node == "interaction" and values.get("gui_message") and isinstance(values["gui_message"], dict):
                    gui = values["gui_message"]
                    if gui.get("message"):
                        final_ans = gui["message"]

        print(f"\n\n🤖 Campus Agent: {final_ans if final_ans else 'Processed successfully.'}")
        print("-" * 50)