# api.py — FastAPI wrapper for BMS Campus Agent
#
# Endpoints:
#   GET  /recognize        → face recognition (stub — ربّطه بالـ face recognition بتاعك)
#   POST /chat             → text message → agent response
#   POST /voice            → audio file → whisper transcription → agent response

import os
import io
import tempfile
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from datetime import datetime, timezone
from dotenv import load_dotenv

load_dotenv(override=True)

# ── استورد الـ orchestrator من main_graph ──────────────────────────────────────
from main_graph import orchestrator, SESSION_HISTORY, BMSGraphState, USER_INFO, CACHED_CONTEXT
from DBs.azure_sql import verify_user_identity, resolve_user, get_lecturer_full_context, get_student_full_context

app = FastAPI(title="BMS Campus Agent API")

# ── CORS عشان الـ React frontend يقدر يكلم الـ API ──────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],        # في الـ production حطّ domain الـ frontend بس
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─────────────────────────────────────────────
# Session store — user_id → SESSION_HISTORY
# (بدل الـ global الواحد، كل user عنده history منفصل)
# ─────────────────────────────────────────────
user_sessions: dict[str, list] = {}

# ─────────────────────────────────────────────
# Schemas
# ─────────────────────────────────────────────
class ChatRequest(BaseModel):
    message:  str
    user_id:  str = "Visitor"   # الـ GUI بيبعت الـ recognized name هنا

class ChatResponse(BaseModel):
    reply: str

class RecognizeResponse(BaseModel):
    status: str   # "recognized" | "unknown"
    name:   str | None = None

# ─────────────────────────────────────────────
# Helper — run the graph for one turn
# ─────────────────────────────────────────────
def _load_user_context(user_id: str):
    """لو الـ user جديد، جيب بياناته وجدوله من الـ DB وكاشّهم."""
    import main_graph
    if main_graph.CACHED_CONTEXT:
        return  # محمّل خلاص

    # resolve الـ user من الـ DB
    try:
        user_info = resolve_user(user_id)
        if user_info.get("role") != "unknown":
            main_graph.USER_INFO.clear()
            main_graph.USER_INFO.update(user_info)
            print(f"[API] Resolved: {user_info.get('name')} ({user_info.get('role')})")
    except Exception as e:
        print(f"[API] resolve_user failed: {e}")
        return

    # جيب الجدول
    try:
        role = main_graph.USER_INFO.get("role", "Instructor")
        if role.lower() == "student":
            data = get_student_full_context(main_graph.USER_INFO.get("id", 0))
        else:
            data = get_lecturer_full_context(main_graph.USER_INFO.get("name", user_id))

        ctx = data if (data and "error" not in data[0]) else []
        main_graph.CACHED_CONTEXT.clear()
        main_graph.CACHED_CONTEXT.extend(ctx)
        print(f"[API] Context loaded: {len(ctx)} sessions for {user_id}")
    except Exception as e:
        print(f"[API] context load failed: {e}")


def run_agent(user_id: str, text: str) -> str:
    import main_graph
    from DBs.azure_sql import resolve_user_name

    # حوّل الـ ID للاسم الحقيقي من الـ DB
    if user_id != "Visitor":
        resolved_name = resolve_user_name(user_id)
    else:
        resolved_name = user_id

    if resolved_name not in user_sessions:
        user_sessions[resolved_name] = []

    if resolved_name != "Visitor":
        _load_user_context(resolved_name)

    main_graph.SESSION_HISTORY = user_sessions[resolved_name]

    initial_state: BMSGraphState = {
        "conversation_id":  f"API_{datetime.now().strftime('%H%M%S%f')}",
        "user_id":          resolved_name,
        "incoming_event": {
            "type":      "voice_input",
            "text":      text,
            "user_id":   resolved_name,
            "timestamp": datetime.now(timezone.utc).isoformat()
        },
        "reports":          [],
        "iteration_count":  0,
        "history":          [],
        "context_data":     main_graph.CACHED_CONTEXT,
        "intent_data":      {},
        "cleaned_text":     "",
        "route":            "",
        "gui_message":      None,
        "pending_question": "",
    }

    final_ans = ""
    try:
        for chunk in orchestrator.stream(initial_state, subgraphs=False):
            for node, values in chunk.items():
                if "reports" in values and values["reports"]:
                    last = values["reports"][-1]
                    if "Response:" in last:
                        final_ans = last.split("Response:", 1)[-1].strip()
                if node == "interaction" and values.get("gui_message") and isinstance(values["gui_message"], dict):
                    gui = values["gui_message"]
                    if gui.get("message"):
                        final_ans = gui["message"]
    except Exception as e:
        print(f"❌ [API Agent Error] {e}")
        return "Sorry, something went wrong. Please try again."

    # حفظ الـ session history بعد الـ run
    user_sessions[resolved_name] = main_graph.SESSION_HISTORY.copy()

    return final_ans or "Processed successfully."

# ─────────────────────────────────────────────
# Endpoints
# ─────────────────────────────────────────────

@app.get("/recognize", response_model=RecognizeResponse)
async def recognize():
    """
    Face recognition endpoint.
    دلوقتي stub — ربّطه بالـ face recognition system بتاعك.
    بيرجع recognized لو فيه وجه معروف، unknown لو لأ.
    """
    # TODO: استبدل الكود ده بالـ face recognition الحقيقي
    # مثال:
    # name = face_recognition_module.recognize()
    # if name:
    #     return {"status": "recognized", "name": name}
    # return {"status": "unknown"}

    # Stub مؤقت — دايماً بيرجع unknown
    return {"status": "unknown", "name": None}


@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    """
    Text chat endpoint.
    الـ GUI بيبعت: { message: "...", user_id: "Ashraf" }
    بيرجع:         { reply: "..." }
    """
    if not req.message.strip():
        raise HTTPException(status_code=400, detail="Message cannot be empty.")

    reply = run_agent(user_id=req.user_id, text=req.message.strip())
    return {"reply": reply}


@app.post("/voice")
async def voice(audio: UploadFile = File(...)):
    """
    Voice endpoint.
    الـ GUI بيبعت audio/webm file.
    بيرجع: { transcription: "...", reply: "..." }

    محتاج: pip install openai-whisper أو openai SDK
    """
    try:
        import whisper   # local whisper
        audio_bytes = await audio.read()

        with tempfile.NamedTemporaryFile(suffix=".webm", delete=False) as tmp:
            tmp.write(audio_bytes)
            tmp_path = tmp.name

        model = whisper.load_model("base")
        result = model.transcribe(tmp_path)
        transcription = result["text"].strip()
        os.unlink(tmp_path)

    except ImportError:
        # لو whisper مش installed، استخدم OpenAI API بدله
        try:
            from openai import OpenAI
            client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            audio_bytes = await audio.read()
            transcription_resp = client.audio.transcriptions.create(
                model="whisper-1",
                file=("recording.webm", audio_bytes, "audio/webm"),
            )
            transcription = transcription_resp.text.strip()
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Whisper error: {e}")

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Audio processing error: {e}")

    # بعد الـ transcription، نبعت للـ agent
    # الـ user_id هنا Visitor لأن الـ GUI مش بيبعت user_id مع الـ voice
    # لو عندك session management ممكن تبعته
    reply = run_agent(user_id="Visitor", text=transcription)

    return {"transcription": transcription, "reply": reply}


# ─────────────────────────────────────────────
# Run
# ─────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=5050, reload=False)