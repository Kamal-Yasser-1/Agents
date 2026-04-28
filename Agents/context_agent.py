import os
import json
from typing import Dict, Any
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage
from DBs.azure_sql import query_schedule_db  # استخدمنا دي عشان هي الأساسية في الـ DBs عندك

# إعداد الموديل المستقر
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash", 
    temperature=0.1, 
    api_key=os.getenv("GOOGLE_API_KEY")
)

def context_agent_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Context Agent Node:
    بمجرد دخول المستخدم، بيسحب بياناته بالكامل من SQL ويبني 'بروفايل لحظي' للأجنتس التانية.
    """
    # 1. تحديد اسم المستخدم (بيجي من الـ Auth أو الـ Interaction)
    user_name = state.get("user_id", state.get("user_name", "Unknown"))
    print(f"🕵️ [Context Agent] Gathering full context for: {user_name}...")

    # 2. الاستعلام من الداتابيز (البحث باسم الدكتور/المحاضر)
    # بنعمل LOWER عشان نضمن إن البحث يلقط "Wafaa" أو "wafaa"
    query = f"SELECT * FROM bms_schedule WHERE LOWER(lecturer) LIKE LOWER('%{user_name}%')"
    
    try:
        db_data = query_schedule_db(query)
        
        if not db_data or (isinstance(db_data, list) and len(db_data) == 0):
            user_context_str = "No schedule found for this user in bms_schedule."
            found_sessions = 0
        else:
            user_context_str = json.dumps(db_data, indent=2, ensure_ascii=False)
            found_sessions = len(db_data)

        # 3. استخدام الـ AI لتحليل البيانات وتجهيز ملخص سريع (Briefing)
        system_prompt = (
            f"You are the BMS Context Assistant. Profile for: {user_name}.\n"
            f"Schedule Data:\n{user_context_str}\n\n"
            "Analyze this schedule. If sessions are found, provide a 1-sentence summary of the next task."
            "If none, say no tasks scheduled."
        )
        
        response = llm.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content="Initialize my session context.")
        ])
        
        summary = response.content

        # 4. تحديث الـ State بالبيانات والملخص
        return {
            "context_data": db_data, 
            "reports": [f"Context initialized: {summary} (Found {found_sessions} entries)"]
        }

    except Exception as e:
        print(f"❌ Context Agent Error: {e}")
        return {
            "context_data": [], 
            "reports": ["Context Agent: Failed to retrieve data from Azure SQL."]
        }