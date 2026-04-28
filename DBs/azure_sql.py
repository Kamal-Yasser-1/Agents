# DBs/azure_sql.py
# ✅ Supports Instructors, Students, and Admins

import requests
from datetime import datetime

BASE_URL = "https://mywebapp2026-bsfxdvevedfsf2d4.germanywestcentral-01.azurewebsites.net/api"

# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────

def _get(endpoint: str) -> list | dict:
    try:
        resp = requests.get(f"{BASE_URL}/{endpoint}", timeout=10)
        resp.raise_for_status()
        return resp.json()
    except requests.exceptions.Timeout:
        return [{"error": "Request timed out."}]
    except requests.exceptions.HTTPError as e:
        return [{"error": f"HTTP Error: {e.response.status_code}"}]
    except Exception as e:
        return [{"error": str(e)}]

def _post(endpoint: str, payload: dict) -> dict:
    try:
        resp = requests.post(f"{BASE_URL}/{endpoint}", json=payload, timeout=10)
        resp.raise_for_status()
        return {"success": True, "data": resp.json() if resp.text else {}}
    except requests.exceptions.HTTPError as e:
        return {"success": False, "error": f"HTTP {e.response.status_code}: {e.response.text}"}
    except Exception as e:
        return {"success": False, "error": str(e)}

def _put(endpoint: str, payload: dict) -> dict:
    try:
        resp = requests.put(f"{BASE_URL}/{endpoint}", json=payload, timeout=10)
        resp.raise_for_status()
        return {"success": True, "data": resp.json() if resp.text else {}}
    except requests.exceptions.HTTPError as e:
        return {"success": False, "error": f"HTTP {e.response.status_code}: {e.response.text}"}
    except Exception as e:
        return {"success": False, "error": str(e)}

def _delete(endpoint: str) -> dict:
    try:
        resp = requests.delete(f"{BASE_URL}/{endpoint}", timeout=10)
        resp.raise_for_status()
        return {"success": True}
    except requests.exceptions.HTTPError as e:
        return {"success": False, "error": f"HTTP {e.response.status_code}: {e.response.text}"}
    except Exception as e:
        return {"success": False, "error": str(e)}

def _fuzzy_match(input_str: str, candidate: str, threshold: int = 80) -> bool:
    """Check if input matches candidate using fuzzy matching."""
    try:
        from thefuzz import fuzz
        return (
            input_str in candidate
            or fuzz.partial_ratio(input_str, candidate) >= threshold
            or fuzz.token_sort_ratio(input_str, candidate) >= threshold
        )
    except ImportError:
        return input_str in candidate

def _clean_name(name: str) -> str:
    """Remove titles like Dr., Eng. for cleaner matching."""
    return (
        name.lower()
        .replace("dr.", "").replace("dr ", "")
        .replace("eng.", "").replace("eng ", "")
        .strip()
    )

def _get_all_users() -> list:
    """Fetch all users from the API."""
    return _get("User")


# ─────────────────────────────────────────────
# AUTH & USER RESOLUTION
# ─────────────────────────────────────────────

def _find_user(user_input: str) -> dict | None:
    """
    Find a user (any role) by name, partial name, fuzzy name, or numeric ID.
    Returns the full user dict or None.
    """
    all_users = _get_all_users()
    if not isinstance(all_users, list) or not all_users or "error" in all_users[0]:
        return None

    input_stripped = user_input.strip()
    input_lower    = input_stripped.lower()
    clean_input    = _clean_name(input_stripped)

    for user in all_users:
        # Match by numeric ID (e.g. 202100191)
        if input_stripped == str(user.get("id", "")).strip():
            return user

        full_name  = user.get("name", "")
        clean_name = _clean_name(full_name)

        if _fuzzy_match(clean_input, clean_name):
            return user

    return None


def verify_user_identity(user_input: str) -> bool:
    """Check if user exists in the system (any role)."""
    try:
        return _find_user(user_input) is not None
    except Exception as e:
        print(f"❌ Auth Error: {e}")
        return False


def resolve_user(user_input: str) -> dict:
    """
    Resolve user input to full user info dict.
    Returns: {id, name, role, ...} or {"name": user_input, "role": "unknown"}
    """
    user = _find_user(user_input)
    if user:
        return user
    return {"name": user_input, "role": "unknown"}


def resolve_user_name(user_input: str) -> str:
    """Return the full name of the matched user."""
    user = _find_user(user_input)
    return user.get("name", user_input) if user else user_input


# ─────────────────────────────────────────────
# CONTEXT — used by main_graph
# ─────────────────────────────────────────────

def get_lecturer_full_context(lecturer_name: str) -> list:
    """Fetch and deduplicate schedule sessions for an instructor."""
    all_schedules = _get("Schedules")
    if not isinstance(all_schedules, list) or (all_schedules and "error" in all_schedules[0]):
        return all_schedules

    clean_input = _clean_name(lecturer_name)
    lecturer_schedules = [
        s for s in all_schedules
        if _fuzzy_match(clean_input, _clean_name(s.get("instructorName", "")))
    ]

    # Deduplicate by (day, time, courseName, roomName) — keep lowest ID
    seen = {}
    for s in lecturer_schedules:
        key = (s.get("day"), s.get("time"), s.get("courseName"), s.get("roomName"))
        if key not in seen or s.get("id", 999999) < seen[key].get("id", 999999):
            seen[key] = s
    lecturer_schedules = list(seen.values())

    day_order = {"Sunday": 1, "Monday": 2, "Tuesday": 3,
                 "Wednesday": 4, "Thursday": 5, "Friday": 6, "Saturday": 7}
    lecturer_schedules.sort(key=lambda s: (
        day_order.get(s.get("day", ""), 8), s.get("time", "")
    ))

    return lecturer_schedules if lecturer_schedules else [{"error": "No sessions found."}]


def get_student_full_context(student_id: int) -> list:
    """
    Fetch schedule for a student based on their enrollments.
    Gets enrolled courses → fetches matching schedules.
    """
    enrollments = _get(f"Enrollments/student/{student_id}")
    if not isinstance(enrollments, list) or not enrollments or "error" in enrollments[0]:
        return [{"error": "No enrollments found."}]

    # Get active enrollments only
    active = [e for e in enrollments if e.get("status", "").lower() == "active"]
    if not active:
        active = enrollments  # fallback: show all if none are "active"

    enrolled_course_ids = {e.get("courseId") for e in active}

    all_schedules = _get("Schedules")
    if not isinstance(all_schedules, list) or (all_schedules and "error" in all_schedules[0]):
        return [{"error": "Could not fetch schedules."}]

    student_schedules = [
        s for s in all_schedules
        if s.get("courseId") in enrolled_course_ids
    ]

    # Deduplicate
    seen = {}
    for s in student_schedules:
        key = (s.get("day"), s.get("time"), s.get("courseName"), s.get("roomName"))
        if key not in seen or s.get("id", 999999) < seen[key].get("id", 999999):
            seen[key] = s
    student_schedules = list(seen.values())

    day_order = {"Sunday": 1, "Monday": 2, "Tuesday": 3,
                 "Wednesday": 4, "Thursday": 5, "Friday": 6, "Saturday": 7}
    student_schedules.sort(key=lambda s: (
        day_order.get(s.get("day", ""), 8), s.get("time", "")
    ))

    return student_schedules if student_schedules else [{"error": "No schedule found for enrolled courses."}]


# ─────────────────────────────────────────────
# DIRECT API HELPERS — used by scheduling_agent tools
# ─────────────────────────────────────────────

def get_all_rooms() -> list:
    return _get("Rooms")

def get_room_schedule(room_id: int) -> list:
    return _get(f"Schedules/room/{room_id}")

def get_all_schedules() -> list:
    return _get("Schedules")

def get_all_courses() -> list:
    return _get("Courses")

def get_course_id(course_name: str) -> int:
    """Find course ID by name using fuzzy matching. Returns 0 if not found."""
    courses = get_all_courses()
    if not isinstance(courses, list):
        return 0
    name_lower = course_name.strip().lower()
    # Exact match first
    for c in courses:
        if c.get("name", "").strip().lower() == name_lower:
            return c.get("id", 0)
    # Partial match fallback
    for c in courses:
        if name_lower in c.get("name", "").strip().lower() or c.get("name", "").strip().lower() in name_lower:
            return c.get("id", 0)
    return 0

def get_all_instructors() -> list:
    return _get("User/Instructors")

def book_schedule(room_id: int, start_time: str, end_time: str,
                  course_id: int, session_type: str, instructor_id: int) -> dict:
    payload = {
        "roomId": room_id,
        "startTime": start_time,
        "endTime": end_time,
        "courseId": course_id,
        "type": session_type,
        "instructorId": instructor_id
    }
    return _post("Schedules", payload)

def delete_schedule(schedule_id: int) -> dict:
    return _delete(f"Schedules/{schedule_id}")

def query_schedule_db(sql_query: str) -> list:
    return get_all_schedules()

def execute_write_query(sql_query: str) -> dict:
    return {"success": False, "error": "Use direct API functions instead."}
