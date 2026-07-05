import os
import re
from typing import TypedDict, List, Dict, Any, Optional, Annotated
import operator
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from models import QueryIntent, JobDescription, CandidateProfile, EmailDraft
import utils
# Define Graph State
class AgentState(TypedDict):
    # Conversational History
    messages: List[Dict[str, str]]
    
    # Recruitment Context
    jd_text: str
    parsed_jd: Optional[Dict[str, Any]]  # Stored as dict for serialization compatibility
    candidates_count: int
    screened_candidates: List[Dict[str, Any]]
    
    # State flags & Memory
    shortlisted_names: List[str]
    finalized_shortlist: List[str]
    pending_action: Optional[Dict[str, Any]]  # {"action": "shortlist", "data": [...]}
    
    # Latest response details
    last_response: str
    error_message: Optional[str]
# 1. ROUTER NODE (CLASSIFIER)
def query_router(state: AgentState) -> str:
    """
    Decides which node to execute next based on user intent.
    Uses regex for simple checks (zero LLM cost) and LLM for complex phrasing.
    """
    if not state["messages"]:
        return "chat"
        
    last_message = state["messages"][-1]["content"].strip()
    last_message_lower = last_message.lower()
    
    # 1. Check if there is a pending action to confirm (yes/y or no/n)
    if state.get("pending_action"):
        if last_message_lower in ["yes", "y", "confirm", "approve", "no", "n", "reject", "cancel"]:
            return "confirm_action"
            
    # 2. Hardcoded regex routing for "How many applicants?" to prevent wasted LLM calls
    count_patterns = [
        r"\bhow many (applicants|candidates|resumes|profiles)\b",
        r"\bcount (applicants|candidates|resumes|profiles)\b",
        r"^applicants count$",
        r"^how many\?$"
    ]
    if any(re.search(pat, last_message_lower) for pat in count_patterns):
        return "count_applicants"
        
    # Load JD/resumes patterns
    load_patterns = [
        r"^load$",
        r"\bload (jd|resumes|data)\b",
        r"start (hiring|process)",
        r"initialize"
    ]
    if any(re.search(pat, last_message_lower) for pat in load_patterns):
        return "load_data"
    # 3. LLM Router for other intents
    # Map intents to graph nodes
    intent_mapping = {
        "load": "load_data",
        "count": "count_applicants",
        "screen": "screen_candidates",
        "rewrite": "rewrite_jd",
        "interview": "generate_questions",
        "salary": "salary_expectations",
        "compare": "compare_candidates",
        "schedule": "schedule_interview",
        "trend": "skill_trend",
        "batch_report": "batch_report",
        "confirm": "confirm_action",
        "chat": "chat"
    }
    
    try:
        # Construct rule-based fallback intent
        fallback_intent = QueryIntent(intent="chat")
        if "how many" in last_message_lower or "count" in last_message_lower:
            fallback_intent = QueryIntent(intent="count")
        elif "top" in last_message_lower or "screen" in last_message_lower or "rank" in last_message_lower:
            fallback_intent = QueryIntent(intent="screen")
        elif "rewrite" in last_message_lower or "optimize" in last_message_lower:
            fallback_intent = QueryIntent(intent="rewrite")
        elif "compare" in last_message_lower or " vs " in last_message_lower or " versus " in last_message_lower:
            fallback_intent = QueryIntent(intent="compare")
        elif "schedule" in last_message_lower or "interview slot" in last_message_lower or "book" in last_message_lower:
            fallback_intent = QueryIntent(intent="schedule")
        elif "trend" in last_message_lower or "in-demand" in last_message_lower or "in demand" in last_message_lower:
            fallback_intent = QueryIntent(intent="trend")
        elif "batch" in last_message_lower or "full report" in last_message_lower or "all candidates" in last_message_lower:
            fallback_intent = QueryIntent(intent="batch_report")
        elif "interview" in last_message_lower or "questions" in last_message_lower:
            cand_name = None
            for word in last_message.split():
                if word[0].isupper() and word.lower() not in ["interview", "questions", "for"]:
                    cand_name = word.strip("?,.!")
                    break
            fallback_intent = QueryIntent(intent="interview", target_candidate=cand_name)
        elif "salary" in last_message_lower or "expectation" in last_message_lower or "pay" in last_message_lower:
            fallback_intent = QueryIntent(intent="salary")
        elif last_message_lower in ["yes", "y", "confirm", "approve"]:
            fallback_intent = QueryIntent(intent="confirm")
        
        # If we have a strong fallback, use it without calling LLM
        if fallback_intent.intent != "chat":
            return intent_mapping.get(fallback_intent.intent, "chat")
        
        # Try LLM classification only as last resort, with aggressive timeout handling
        prompt = f"""Classify the user's latest query into one of these intents:
- 'load': if the user wants to load/parse a Job Description or initialize resumes (e.g. 'here is the JD', 'load resumes', 'start hiring process').
- 'screen': if the user wants to screen candidates or find the top matching profiles (e.g. 'get me top candidates', 'who matches best', 'rank applicants').
- 'rewrite': if the user wants to rewrite/improve/update the Job Description (e.g. 'rewrite this JD for a startup', 'optimize the job post').
- 'interview': if the user wants interview questions for a candidate (e.g. 'generate questions for John Doe', 'interview questions for Jane').
- 'salary': if the user wants salary expectations or market benchmarks (e.g. 'what is the salary for this role?', 'salary expectations?').
- 'compare': if the user wants a side-by-side comparison of two or more candidates (e.g. 'compare John and Jane', 'John vs Jane').
- 'schedule': if the user wants to book/propose an interview slot for a candidate (e.g. 'schedule an interview with John', 'book interview for Jane').
- 'trend': if the user wants to know which skills are currently in-demand for the role (e.g. 'what skills are trending?', 'in-demand skills for this role').
- 'batch_report': if the user wants a full report on ALL screened candidates rather than just the top few (e.g. 'give me the full report', 'show all candidates').
- 'confirm': if the user is confirming or finalizing a shortlist or scheduled action (e.g. 'finalize this shortlist', 'confirm these candidates', 'yes').
- 'chat': if it is general greeting or conversation (e.g. 'hello', 'who are you', 'thank you').
Query: "{last_message}"
"""
        try:
            classification = utils.safe_llm_invoke(prompt, structured_schema=QueryIntent, fallback_value=fallback_intent)
            intent = classification.intent
        except (KeyboardInterrupt, TimeoutError, Exception):
            # On any error, use fallback
            intent = fallback_intent.intent
        
        return intent_mapping.get(intent, "chat")
    except Exception as e:
        # Fall back to general chat node on error
        return "chat"
# 2. NODE IMPLEMENTATIONS
def load_data_node(state: AgentState) -> Dict[str, Any]:
    """Loads and parses the default JD and indexes resumes if not already done."""
    messages = state["messages"]
    last_msg = messages[-1]["content"] if messages else ""
    
    # 1. Parse/Index resumes
    count, warnings = utils.parse_and_index_resumes()
    
    # 2. Parse JD
    # Check if user provided a new JD text in the message
    if "role:" in last_msg.lower() or "description:" in last_msg.lower() or len(last_msg) > 100:
        jd_text = last_msg
    else:
        # Load default JD file
        jd_path = "data/jds/ai_engineer_jd.txt"
        if os.path.exists(jd_path):
            with open(jd_path, "r") as f:
                jd_text = f.read()
        else:
            jd_text = "Role: Senior AI Engineer\nRequired Skills: Python, LangChain, LangGraph\nExperience: 4 years"
            
    try:
        parsed_jd_obj = utils.parse_jd(jd_text)
        parsed_jd_dict = parsed_jd_obj.model_dump()
        jd_suggestions = utils.suggest_jd_improvements(parsed_jd_obj)
        response = (
            f"Successfully parsed Job Description for **{parsed_jd_obj.role_name}**!\n"
            f"- **Required Experience**: {parsed_jd_obj.required_experience_years}+ years\n"
            f"- **Required Skills**: {', '.join(parsed_jd_obj.required_skills)}\n\n"
            f"Indexed **{count}** candidate resumes into ChromaDB vector store.\n\n"
            f"**JD Quality Check:**\n" + "\n".join(f"- {s}" for s in jd_suggestions)
        )
        return {
            "jd_text": jd_text,
            "parsed_jd": parsed_jd_dict,
            "candidates_count": count,
            "last_response": response,
            "error_message": None
        }
    except Exception as e:
        return {
            "last_response": f"Error parsing Job Description: {str(e)}",
            "error_message": str(e)
        }
def count_applicants_node(state: AgentState) -> Dict[str, Any]:
    """Plain python count of loaded resumes (proves smart routing / no LLM call)."""
    count = 0
    resume_dir = "data/resumes"
    if os.path.exists(resume_dir):
        count = len([f for f in os.listdir(resume_dir) if f.endswith(".txt")])
        
    response = f"There are currently **{count}** applicants loaded in the recruitment system database."
    return {
        "candidates_count": count,
        "last_response": response
    }
def screen_candidates_node(state: AgentState) -> Dict[str, Any]:
    """RAG screening over candidates based on the loaded JD."""
    if not state.get("parsed_jd"):
        # Auto-load if not already parsed
        load_state = load_data_node(state)
        state.update(load_state)
        if not state.get("parsed_jd"):
            return {"last_response": "Please load a Job Description first before screening candidates."}
            
    # Re-construct Pydantic model for internal utility use
    jd = JobDescription(**state["parsed_jd"])
    
    # Run screening search and compute scores
    screened = utils.screen_candidates(jd)
    
    if not screened:
        return {
            "screened_candidates": [],
            "last_response": "No matching candidates found in the vector database."
        }
        
    # Pick top 3 candidates for shortlisting proposal
    top_3 = [c["name"] for c in screened[:3]]
    
    # Format the top candidates as a markdown table
    response = f"### RAG Candidate Screening Results (Role: {jd.role_name})\n\n"
    response += "| Rank | Candidate Name | Experience | Match Score | Key Gaps |\n"
    response += "| :--- | :--- | :--- | :--- | :--- |\n"
    for idx, c in enumerate(screened[:5], 1):
        gaps = ", ".join(c["missing_skills"]) if c["missing_skills"] else "None"
        response += f"| #{idx} | **{c['name']}** | {c['experience_years']} years | {c['match_score']}% | {gaps} |\n"
        
    response += "\n" + utils.compute_jd_mismatch_stats(jd, screened) + "\n"
    response += f"\nBased on alignment, I suggest shortlisting these top candidates: **{', '.join(top_3)}**.\n"
    response += "Would you like me to finalize this shortlist? (yes/no)"
    
    return {
        "screened_candidates": screened,
        "pending_action": {"action": "finalize_shortlist", "data": top_3},
        "last_response": response
    }
def rewrite_jd_node(state: AgentState) -> Dict[str, Any]:
    """Rewrites the current job description with custom styling/tone."""
    if not state.get("parsed_jd"):
        load_state = load_data_node(state)
        state.update(load_state)
        if not state.get("parsed_jd"):
            return {"last_response": "Please load a Job Description first before rewriting it."}
            
    jd = JobDescription(**state["parsed_jd"])
    
    last_msg = state["messages"][-1]["content"].lower()
    
    # Extract tone instructions
    tone = "startup"
    if "corporate" in last_msg or "enterprise" in last_msg:
        tone = "corporate"
    elif "casual" in last_msg or "informal" in last_msg:
        tone = "casual"
        
    rewritten = utils.rewrite_jd_llm(jd, tone=tone)
    response = f"### Rewritten Job Description ({tone.upper()} Tone):\n\n```markdown\n{rewritten}\n```"
    return {
        "last_response": response
    }
def generate_questions_node(state: AgentState) -> Dict[str, Any]:
    """Generates customized interview questions grounded in JD and candidate gaps."""
    if not state.get("parsed_jd"):
        return {"last_response": "Please load a Job Description first."}
    if not state.get("screened_candidates"):
        # Auto screen to get candidate lists
        screen_state = screen_candidates_node(state)
        state.update(screen_state)
        if not state.get("screened_candidates"):
            return {"last_response": "No candidates available to generate questions for."}
            
    jd = JobDescription(**state["parsed_jd"])
    last_msg = state["messages"][-1]["content"]
    
    # Extract candidate name from query using LLM
    try:
        # Rule-based fallback: try to find matching name in text
        fallback_name = "None"
        for c in state["screened_candidates"]:
            if c["name"].lower() in last_msg.lower():
                fallback_name = c["name"]
                break
                
        prompt = f"""Identify the candidate name from this query: "{last_msg}".
Available Candidates: {', '.join([c['name'] for c in state['screened_candidates']])}
If no candidate matches, respond exactly with 'None'. Otherwise respond ONLY with the candidate's full name.
"""
        cand_name_res = utils.safe_llm_invoke(prompt, fallback_value=fallback_name)
        cand_name = cand_name_res.content.strip() if hasattr(cand_name_res, 'content') else str(cand_name_res).strip()
    except Exception:
        cand_name = "None"
        
    # Match candidate in screened list
    target_cand = None
    if cand_name != "None":
        for c in state["screened_candidates"]:
            if cand_name.lower() in c["name"].lower() or c["name"].lower() in cand_name.lower():
                target_cand = c
                break
                
    if not target_cand:
        # Default to the top candidate if none specified
        target_cand = state["screened_candidates"][0]
        
    questions = utils.generate_questions_llm(jd, target_cand)
    resume_text = utils.get_resume_full_text(target_cand["filename"])
    red_flags = utils.detect_resume_red_flags(resume_text, target_cand["experience_years"])
    
    response = f"### Interview Prep for **{target_cand['name']}**\n"
    response += f"Gaps identified: {', '.join(target_cand['missing_skills']) if target_cand['missing_skills'] else 'None'}\n\n"
    response += "**Customized Interview Questions (Grounded in JD + Resume Gaps):**\n"
    for i, q in enumerate(questions, 1):
        response += f"{i}. {q}\n"
    response += f"\n**Resume Integrity Check:** {' '.join(red_flags)}\n"
        
    return {
        "last_response": response
    }
def salary_expectations_node(state: AgentState) -> Dict[str, Any]:
    """Searches Tavily (or local cache fallback) for role salary expectations."""
    if not state.get("parsed_jd"):
        role = "AI Engineer"
    else:
        role = state["parsed_jd"]["role_name"]
        
    # Look up benchmark
    benchmark = utils.get_salary_benchmark(role)
    
    response = f"### Salary Expectation Benchmark (Role: {role})\n"
    response += f"- **Average Salary**: ₹{benchmark.get('avg_lakhs')} Lakhs CTC\n"
    response += f"- **Range**: ₹{benchmark.get('min_lakhs')} Lakhs - ₹{benchmark.get('max_lakhs')} Lakhs CTC\n"
    response += f"- **Currency**: {benchmark.get('currency', 'INR')}\n"
    response += f"- **Sources**: {', '.join(benchmark.get('sources', []))}\n"
    
    return {
        "last_response": response
    }
def confirm_action_node(state: AgentState) -> Dict[str, Any]:
    """Handles Human-in-the-loop confirmation response."""
    last_msg = state["messages"][-1]["content"].strip().lower()
    pending = state.get("pending_action")
    
    if not pending:
        return {
            "last_response": "There are no pending actions to confirm.",
            "pending_action": None
        }
        
    action_type = pending["action"]
    data = pending["data"]
    
    confirmed = last_msg in ["yes", "y", "confirm", "approve"]
    
    if action_type == "finalize_shortlist":
        if confirmed:
            finalized = data
            response = "✅ **Shortlist Finalized Successfully!**\n"
            response += f"Confirmed Candidates: **{', '.join(finalized)}**\n\n"
            
            # Auto-draft (and mock-send) candidate acceptance emails for the shortlisted profiles
            response += "### Drafted & Sent Candidate Emails (simulated — no SMTP/Gmail configured)\n"
            jd = JobDescription(**state["parsed_jd"])
            for name in finalized:
                cand_info = next((c for c in state["screened_candidates"] if c["name"] == name), None)
                if cand_info:
                    draft = utils.draft_email_llm(cand_info, approved=True, jd=jd)
                    utils.send_email_mock(draft)
                    response += f"\n--- \n**To**: {draft.recipient_email}\n**Subject**: {draft.subject}\n\n{draft.body}\n"
            
            return {
                "finalized_shortlist": finalized,
                "pending_action": None,
                "last_response": response
            }
        else:
            return {
                "pending_action": None,
                "last_response": "❌ **Shortlist confirmation cancelled.** The shortlist was not finalized."
            }
    elif action_type == "schedule_interview":
        if confirmed:
            cand_name = data["candidate"]
            slot = data["slot"]
            utils.log_scheduled_interview(cand_name, slot)
            response = (
                f"✅ **Interview Scheduled!**\n"
                f"Candidate: **{cand_name}**\nSlot: **{slot}**\n"
                f"(Booked to mock calendar: data/interview_calendar.json)"
            )
            return {"pending_action": None, "last_response": response}
        else:
            return {
                "pending_action": None,
                "last_response": "❌ **Interview scheduling cancelled.** No slot was booked."
            }
    else:
        return {
            "pending_action": None,
            "last_response": "There was a pending action but its type wasn't recognized, so it was cleared."
        }
def chat_node(state: AgentState) -> Dict[str, Any]:
    """Handles general conversational chatbot responses."""
    last_msg = state["messages"][-1]["content"] if state["messages"] else ""
    last_msg_lower = last_msg.lower()
    
    # Provide local responses for common greetings/questions without calling LLM
    local_responses = {
        "hello": "👋 Welcome to HireGraph AI! I'm your recruitment assistant. Try asking: 'How many applicants?', 'Get me top candidates', or 'Compare John Doe and Jane Smith'.",
        "hi": "👋 Welcome to HireGraph AI! Try: 'Get me top candidates' or 'How many applicants?'",
        "help": "I can help you with: 'How many applicants?', 'Get me top candidates', 'Interview questions for [Name]', 'Compare candidates', 'Salary expectations', 'What skills are trending?'",
        "who are you": "I'm HireGraph AI, a recruitment assistant built with LangChain and ChromaDB. I help screen resumes, generate interview questions, and analyze candidate fit.",
        "thank you": "You're welcome! Is there anything else I can help with?",
        "thanks": "Happy to help! Need anything else?",
        "how are you": "I'm running great! Ready to help you manage this hiring process. What would you like to do?",
        "what can you do": "I can: screen candidates, generate interview questions, rewrite JDs, compare candidates, check salary expectations, and more. Try 'Get me top candidates'!"
    }
    
    # Check for exact or close matches
    for key, response in local_responses.items():
        if key in last_msg_lower or last_msg_lower in key:
            return {"last_response": response}
    
    # If no local match, try LLM but with fallback
    history = []
    for m in state["messages"][-5:]:
        role = "User" if m["role"] == "user" else "Assistant"
        history.append(f"{role}: {m['content']}")
        
    prompt = f"""You are a professional HR Recruitment Assistant. Help recruiters manage hiring, write JDs, screen resumes, and look up salaries.
Chat History:
{chr(10).join(history)}
Generate a helpful, concise, professional response. If appropriate, suggest recruitment tasks like 'Get me top candidates', 'How many applicants?', 'Rewrite this JD', or 'Salary expectations for this role?'.
"""
    try:
        response_res = utils.safe_llm_invoke(prompt)
        response = response_res.content.strip() if hasattr(response_res, 'content') else str(response_res).strip()
    except:
        # Fallback if LLM unavailable
        response = "I'm here to help with recruitment tasks! Try: 'Get me top candidates', 'How many applicants?', or 'Compare candidates'."
    
    return {
        "last_response": response
    }
def compare_candidates_node(state: AgentState) -> Dict[str, Any]:
    """Nice-to-have: side-by-side comparison of two or more screened candidates."""
    if not state.get("parsed_jd"):
        load_state = load_data_node(state)
        state.update(load_state)
    if not state.get("screened_candidates"):
        screen_state = screen_candidates_node(state)
        state.update(screen_state)
        if not state.get("screened_candidates"):
            return {"last_response": "Please screen candidates first before comparing them."}
            
    last_msg = state["messages"][-1]["content"]
    jd = JobDescription(**state["parsed_jd"])
    candidate_names = [c["name"] for c in state["screened_candidates"]]
    fallback_names = [n for n in candidate_names if n.lower() in last_msg.lower()]
    
    try:
        prompt = f"""Identify which candidate names from this list are mentioned in the query: "{last_msg}".
Available Candidates: {', '.join(candidate_names)}
Respond ONLY with a comma-separated list of the exact matching names, or 'None' if none are mentioned.
"""
        class _Fallback:
            content = ", ".join(fallback_names) if fallback_names else "None"
        res = utils.safe_llm_invoke(prompt, fallback_value=_Fallback())
        text = res.content.strip() if hasattr(res, "content") else str(res).strip()
        names = [n.strip() for n in text.split(",")] if text.lower() != "none" and text else fallback_names
    except Exception:
        names = fallback_names
        
    if len(names) < 2:
        names = candidate_names[:2]  # default to the top 2 screened candidates
        
    table = utils.compare_candidates(names, state["screened_candidates"], jd)
    if not table:
        return {"last_response": "Couldn't find at least 2 matching candidates to compare. Try naming two screened candidates."}
        
    return {"last_response": f"### Candidate Comparison\n\n{table}\n"}
def skill_trend_node(state: AgentState) -> Dict[str, Any]:
    """Nice-to-have: live (Tavily) skill trend analysis diffed against the current JD."""
    if not state.get("parsed_jd"):
        load_state = load_data_node(state)
        state.update(load_state)
        if not state.get("parsed_jd"):
            return {"last_response": "Please load a Job Description first before checking skill trends."}
            
    jd = JobDescription(**state["parsed_jd"])
    trend = utils.get_skill_trend_analysis(jd)
    
    response = f"### Skill Trend Analysis for **{jd.role_name}**\n\n"
    response += f"**Trending Skills (Market):** {', '.join(trend['trending_skills'])}\n\n"
    response += f"**Already Covered in your JD:** {', '.join(trend['covered_in_jd']) or 'None'}\n"
    response += f"**Trending but Missing from JD:** {', '.join(trend['missing_from_jd']) or 'None — your JD is current!'}\n\n"
    response += f"*Sources: {', '.join(trend['sources'])}*"
    
    return {"last_response": response}
def batch_report_node(state: AgentState) -> Dict[str, Any]:
    """Moon-shot: full batch screening report across ALL candidates (not just the top few)."""
    if not state.get("parsed_jd"):
        load_state = load_data_node(state)
        state.update(load_state)
    if not state.get("screened_candidates"):
        screen_state = screen_candidates_node(state)
        state.update(screen_state)
        if not state.get("screened_candidates"):
            return {"last_response": "No candidates available for a batch report."}
            
    jd = JobDescription(**state["parsed_jd"])
    screened = state["screened_candidates"]
    
    response = f"### Full Batch Screening Report ({len(screened)} Candidates, Role: {jd.role_name})\n\n"
    response += "| Rank | Candidate | Experience | Match Score | Missing Skills |\n"
    response += "| :--- | :--- | :--- | :--- | :--- |\n"
    for idx, c in enumerate(screened, 1):
        gaps = ", ".join(c["missing_skills"]) if c["missing_skills"] else "None"
        response += f"| #{idx} | {c['name']} | {c['experience_years']} yrs | {c['match_score']}% | {gaps} |\n"
    response += "\n" + utils.compute_jd_mismatch_stats(jd, screened)
    
    return {"last_response": response}
def schedule_interview_node(state: AgentState) -> Dict[str, Any]:
    """Nice-to-have: mock interview scheduling with human-in-the-loop confirmation."""
    if not state.get("screened_candidates"):
        screen_state = screen_candidates_node(state)
        state.update(screen_state)
        if not state.get("screened_candidates"):
            return {"last_response": "Please screen candidates first before scheduling interviews."}
            
    last_msg = state["messages"][-1]["content"]
    candidates = state["screened_candidates"]
    target = next((c for c in candidates if c["name"].lower() in last_msg.lower()), None)
    if not target:
        target = candidates[0]  # default to the top-ranked candidate
        
    slots = utils.generate_interview_slots(target["name"])
    proposed = slots[0]
    
    response = f"### Interview Scheduling — {target['name']}\n"
    response += f"Proposed slot: **{proposed}**\n"
    response += f"Other available slots: {', '.join(slots[1:])}\n\n"
    response += "Shall I book the proposed slot? (yes/no)"
    
    return {
        "pending_action": {"action": "schedule_interview", "data": {"candidate": target["name"], "slot": proposed}},
        "last_response": response
    }
# 3. BUILD GRAPH
builder = StateGraph(AgentState)
# Add Nodes
builder.add_node("load_data", load_data_node)
builder.add_node("count_applicants", count_applicants_node)
builder.add_node("screen_candidates", screen_candidates_node)
builder.add_node("rewrite_jd", rewrite_jd_node)
builder.add_node("generate_questions", generate_questions_node)
builder.add_node("salary_expectations", salary_expectations_node)
builder.add_node("compare_candidates", compare_candidates_node)
builder.add_node("skill_trend", skill_trend_node)
builder.add_node("batch_report", batch_report_node)
builder.add_node("schedule_interview", schedule_interview_node)
builder.add_node("confirm_action", confirm_action_node)
builder.add_node("chat", chat_node)
# Connect edges using Router
builder.add_conditional_edges(
    START,
    query_router,
    {
        "load_data": "load_data",
        "count_applicants": "count_applicants",
        "screen_candidates": "screen_candidates",
        "rewrite_jd": "rewrite_jd",
        "generate_questions": "generate_questions",
        "salary_expectations": "salary_expectations",
        "compare_candidates": "compare_candidates",
        "skill_trend": "skill_trend",
        "batch_report": "batch_report",
        "schedule_interview": "schedule_interview",
        "confirm_action": "confirm_action",
        "chat": "chat"
    }
)
# Every node transitions directly to END, since the router evaluates every user input turn as a new START
ALL_NODES = [
    "load_data", "count_applicants", "screen_candidates", "rewrite_jd", "generate_questions",
    "salary_expectations", "compare_candidates", "skill_trend", "batch_report",
    "schedule_interview", "confirm_action", "chat"
]
for node in ALL_NODES:
    builder.add_edge(node, END)
# Compile Agent with checkpointer for conversation memory
memory = MemorySaver()
recruitment_agent = builder.compile(checkpointer=memory)