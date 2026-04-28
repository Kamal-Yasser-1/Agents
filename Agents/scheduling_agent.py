# Agents/scheduling_agent.py
import os
from datetime import datetime
from langgraph.graph import StateGraph, START, MessagesState
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from DBs.azure_sql import (
    get_all_schedules,
    get_room_schedule,
    get_all_rooms,
    get_all_instructors,
    book_schedule,
    delete_schedule,
    get_course_id,
)


# ─────────────────────────────────────────────
# TOOLS
# ─────────────────────────────────────────────

def check_schedule(instructor_name: str) -> list:
    """
    Get all schedule sessions for a specific instructor by name.
    Returns sessions with room, course, day, and time info.
    Args:
        instructor_name: Full or partial name e.g. 'Wafaa' or 'Wafaa Rady'
    """
    all_schedules = get_all_schedules()
    if not isinstance(all_schedules, list):
        return [{"error": "Failed to fetch schedules"}]

    name_lower = instructor_name.strip().lower()
    results = [
        s for s in all_schedules
        if name_lower in s.get("instructorName", "").lower()
    ]

    if not results:
        return [{"message": f"No sessions found for {instructor_name}."}]

    day_order = {
        "Sunday": 1, "Monday": 2, "Tuesday": 3,
        "Wednesday": 4, "Thursday": 5, "Friday": 6, "Saturday": 7
    }
    results.sort(key=lambda s: (day_order.get(s.get("day", ""), 8), s.get("time", "")))
    return results


def check_room_availability(room_name: str, date: str, start_time: str, end_time: str) -> dict:
    """
    Check if a room is available on a specific date and time range.
    Args:
        room_name: Room name e.g. 'A101', 'B202'
        date: Date 'YYYY-MM-DD'
        start_time: Start time 'HH:MM' 24-hour
        end_time: End time 'HH:MM' 24-hour
    """
    rooms = get_all_rooms()
    if not isinstance(rooms, list):
        return {"available": False, "error": "Could not fetch rooms."}

    room = next(
        (r for r in rooms if r.get("roomName", "").lower() == room_name.strip().lower()),
        None
    )
    if not room:
        available_names = [r.get("roomName") for r in rooms]
        return {"available": False, "error": f"Room '{room_name}' not found. Available: {available_names}"}

    room_id = room["id"]
    room_schedules = get_room_schedule(room_id)

    if not isinstance(room_schedules, list):
        return {"available": True, "room_id": room_id, "conflicts": []}

    req_start = datetime.strptime(f"{date}T{start_time}", "%Y-%m-%dT%H:%M")
    req_end   = datetime.strptime(f"{date}T{end_time}",   "%Y-%m-%dT%H:%M")

    conflicts = []
    for s in room_schedules:
        try:
            s_start = datetime.fromisoformat(s.get("startTime", ""))
            s_end   = datetime.fromisoformat(s.get("endTime",   ""))
            if s_start < req_end and s_end > req_start:
                conflicts.append({
                    "course":     s.get("courseName", "Unknown"),
                    "instructor": s.get("instructorName", "Unknown"),
                    "time":       s.get("time", ""),
                    "day":        s.get("day", "")
                })
        except Exception:
            continue

    return {
        "available": len(conflicts) == 0,
        "room_id":   room_id,
        "room_name": room_name,
        "conflicts": conflicts
    }


def book_room(instructor_name: str, room_name: str, course_name: str,
              date: str, start_time: str, end_time: str) -> dict:
    """
    Book a room for an extra session after checking availability.
    Args:
        instructor_name: Full name of the instructor
        room_name: Room name e.g. 'A101'
        course_name: Name of the session or course
        date: Date 'YYYY-MM-DD'
        start_time: Start time 'HH:MM' 24-hour
        end_time: End time 'HH:MM' 24-hour
    """
    availability = check_room_availability(room_name, date, start_time, end_time)

    if not availability.get("available"):
        if availability.get("error"):
            return {"success": False, "message": availability["error"]}
        conflicts = availability.get("conflicts", [])
        conflict_info = ""
        if conflicts:
            c = conflicts[0]
            conflict_info = f" Conflict with '{c.get('course')}' ({c.get('time')})."
        return {"success": False, "message": f"Room {room_name} is NOT available on {date} from {start_time} to {end_time}.{conflict_info}"}

    room_id = availability["room_id"]

    instructors = get_all_instructors()
    instructor = next(
        (i for i in (instructors if isinstance(instructors, list) else [])
         if instructor_name.strip().lower() in i.get("name", "").lower()),
        None
    )
    instructor_id = instructor["id"] if instructor else 0

    course_id = get_course_id(course_name)

    result = book_schedule(
        room_id=room_id,
        start_time=f"{date}T{start_time}:00",
        end_time=f"{date}T{end_time}:00",
        course_id=course_id,
        session_type="Extra",
        instructor_id=instructor_id
    )

    if result.get("success"):
        return {"success": True, "message": (
            f"✅ Booking confirmed!\n"
            f"   Room      : {room_name}\n"
            f"   Session   : {course_name}\n"
            f"   Date      : {date}\n"
            f"   Time      : {start_time} - {end_time}\n"
            f"   Instructor: {instructor_name}"
        )}
    return {"success": False, "message": f"Booking failed: {result.get('error', 'Unknown error')}"}


def cancel_booking(schedule_id: int) -> dict:
    """
    Cancel an existing booking by its schedule ID.
    Args:
        schedule_id: The numeric ID of the schedule entry to cancel
    """
    result = delete_schedule(schedule_id)
    if result.get("success"):
        return {"success": True, "message": f"✅ Booking #{schedule_id} cancelled successfully."}
    return {"success": False, "message": f"Cancellation failed: {result.get('error', 'Unknown error')}"}


def get_my_bookings(instructor_name: str) -> list:
    """
    Get all upcoming sessions for an instructor.
    Args:
        instructor_name: Full or partial name of the instructor
    """
    all_schedules = get_all_schedules()
    if not isinstance(all_schedules, list):
        return [{"error": "Failed to fetch schedules"}]

    now = datetime.now()
    name_lower = instructor_name.strip().lower()

    upcoming = []
    for s in all_schedules:
        if name_lower not in s.get("instructorName", "").lower():
            continue
        try:
            if datetime.fromisoformat(s.get("startTime", "2000-01-01")) >= now:
                upcoming.append(s)
        except Exception:
            continue

    if not upcoming:
        return [{"message": f"No upcoming sessions found for {instructor_name}."}]

    upcoming.sort(key=lambda s: s.get("startTime", ""))
    return upcoming


def get_available_rooms(date: str, start_time: str, end_time: str) -> list:
    """
    Find all rooms that are free during a specific time slot.
    Args:
        date: Date 'YYYY-MM-DD'
        start_time: Start time 'HH:MM' 24-hour
        end_time: End time 'HH:MM' 24-hour
    """
    rooms = get_all_rooms()
    if not isinstance(rooms, list):
        return [{"error": "Could not fetch rooms."}]

    available = []
    for room in rooms:
        result = check_room_availability(room.get("roomName", ""), date, start_time, end_time)
        if result.get("available"):
            available.append({
                "roomName": room.get("roomName"),
                "floor":    room.get("floor"),
                "capacity": room.get("capacity"),
                "id":       room.get("id")
            })

    return available if available else [{"message": "No rooms available in that time slot."}]


# ─────────────────────────────────────────────
# LLM
# ─────────────────────────────────────────────

def add_lecture(instructor_name: str, room_name: str, course_name: str,
                day: str, start_time: str, end_time: str) -> dict:
    """
    Add a new recurring lecture to the schedule (permanent, not a one-time booking).
    Use this when the instructor wants to add a new session to their weekly schedule.
    Args:
        instructor_name: Full name of the instructor
        room_name: Room name e.g. 'A101'
        course_name: Course name
        day: Day of week e.g. 'Monday', 'Tuesday'
        start_time: Start time 'HH:MM' 24-hour
        end_time: End time 'HH:MM' 24-hour
    """
    from datetime import datetime, timedelta

    # Calculate next occurrence of the given day
    days_map = {"Sunday": 6, "Monday": 0, "Tuesday": 1, "Wednesday": 2,
                "Thursday": 3, "Friday": 4, "Saturday": 5}
    today      = datetime.now()
    target_day = days_map.get(day.capitalize(), 0)
    days_ahead = (target_day - today.weekday()) % 7
    if days_ahead == 0:
        days_ahead = 7
    next_date = (today + timedelta(days=days_ahead)).strftime("%Y-%m-%d")

    # Check availability first
    availability = check_room_availability(room_name, next_date, start_time, end_time)
    if not availability.get("available"):
        if availability.get("error"):
            return {"success": False, "message": availability["error"]}
        conflicts = availability.get("conflicts", [])
        conflict_info = f" Conflict with '{conflicts[0].get('course')}' ({conflicts[0].get('time')})." if conflicts else ""
        return {"success": False, "message": f"Room {room_name} is NOT available on {day} {start_time}-{end_time}.{conflict_info}"}

    room_id = availability["room_id"]

    # Find instructor
    instructors = get_all_instructors()
    instructor  = next(
        (i for i in (instructors if isinstance(instructors, list) else [])
         if instructor_name.strip().lower() in i.get("name", "").lower()),
        None
    )
    instructor_id = instructor["id"] if instructor else 0

    course_id = get_course_id(course_name)

    result = book_schedule(
        room_id=room_id,
        start_time=f"{next_date}T{start_time}:00",
        end_time=f"{next_date}T{end_time}:00",
        course_id=course_id,
        session_type="Lecture",
        instructor_id=instructor_id
    )

    if result.get("success"):
        return {"success": True, "message": (
            f"✅ Lecture added to schedule!\n"
            f"   Course    : {course_name}\n"
            f"   Day       : {day}\n"
            f"   Time      : {start_time} - {end_time}\n"
            f"   Room      : {room_name}\n"
            f"   Instructor: {instructor_name}"
        )}
    return {"success": False, "message": f"Failed to add lecture: {result.get('error', 'Unknown error')}"}


tools = [
    check_schedule,
    check_room_availability,
    book_room,
    add_lecture,
    cancel_booking,
    get_my_bookings,
    get_available_rooms,
]

llm_base = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    api_key=os.getenv("GOOGLE_API_KEY")
)
llm = llm_base.bind_tools(tools)

SYSTEM_PROMPT = """
You are Campus Agent, a smart assistant for university building management.

YOU CAN HELP WITH:
1. Checking lecture schedules (for instructors AND students)
2. Booking rooms for extra/one-time sessions
3. Adding new recurring lectures to an instructor's schedule
4. Checking room availability before booking
5. Cancelling existing bookings by schedule ID
6. Showing upcoming sessions
7. Finding available rooms for a time slot

USER ROLES:
- Instructor → can check schedule, book rooms, add lectures, cancel bookings
- Student    → can only check their own schedule (read-only)
- Admin      → full access

BOOKING RULES:
- ALWAYS call check_room_availability BEFORE calling book_room
- date format: 'YYYY-MM-DD'
- start_time / end_time format: 'HH:MM' 24-hour
- Example: '2:00 PM' -> '14:00' | '23/4/2026' -> '2026-04-23'
- Day name only (e.g. 'next Sunday') -> calculate the actual upcoming date
- If you have ALL required info -> proceed directly without asking
- Only ask for missing info if truly needed
- COURSE RULE: Each session in the pre-loaded schedule has a [courseId:X] field.
  When booking, ALWAYS use the courseId from the matching course in the schedule.
  Match the user's mentioned course name to the closest one in their schedule.
  If no match found, list the instructor's courses and ask them to pick one.
  NEVER use courseId:0 — always use the actual ID from the schedule.

SCHEDULE RULES:
- The instructor's pre-loaded schedule is shown above — use it FIRST before calling any tool
- 'Today' = the current day shown above
- If asked about today's schedule and it exists in the pre-loaded data -> answer directly, no tool call needed
- instructorName in the data contains full names like 'Dr Ahmed Hassan'

LANGUAGE:
- Detect user language and respond in SAME language
- Arabic -> Arabic | English -> English

RESPONSE FORMAT:
- Schedules: clean table or bullet list showing day, time, course, room
- Bookings: confirm all details clearly
- Be concise and direct — never ask for info you already have
"""


def _extract_text(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict):
                parts.append(item.get("text", str(item)))
            else:
                parts.append(str(item))
        return " ".join(parts)
    return str(content)


def assistant(state: dict):
    from langchain_core.messages import ToolMessage, SystemMessage

    messages = state.get("messages", [])
    if not messages:
        return {"messages": [AIMessage(content="Please type your question.")]}

    # Gemini rejects SystemMessages and orphan ToolMessages.
    # AIMessage with empty content but tool_calls MUST be kept for the tool cycle.
    def has_tool_calls(msg):
        return (
            bool(getattr(msg, "tool_calls", None)) or
            bool(msg.additional_kwargs.get("tool_calls")) or
            bool(msg.additional_kwargs.get("function_call"))
        )

    clean = []
    for msg in messages:
        if isinstance(msg, SystemMessage):
            continue
        c = msg.content
        is_empty = (isinstance(c, str) and not c.strip()) or (isinstance(c, list) and not c)
        if is_empty and isinstance(msg, AIMessage):
            clean.append(msg)  # keep even if empty — carries tool_calls
            continue
        if is_empty:
            continue
        clean.append(msg)

    # Drop orphan ToolMessages (no preceding AIMessage with tool_calls)
    valid = []
    for msg in clean:
        if isinstance(msg, ToolMessage):
            prev = valid[-1] if valid else None
            if not (prev and isinstance(prev, AIMessage) and has_tool_calls(prev)):
                continue
        valid.append(msg)

    if not valid:
        return {"messages": [AIMessage(content="Please type your question.")]}

    response = llm.invoke(valid)
    return {"messages": [response]}


builder = StateGraph(MessagesState)
builder.add_node("assistant", assistant)
builder.add_node("tools", ToolNode(tools))
builder.add_edge(START, "assistant")
builder.add_conditional_edges("assistant", tools_condition)
builder.add_edge("tools", "assistant")

scheduling_agent = builder.compile()