# HireGraph AI — Enhanced Recruitment Chatbot

This is your original Day-4 hackathon submission, audited against the brief and upgraded to cover
every **Must Have**, most **Nice to Have**, and two **Moon-shot** items, while fixing two real bugs
that would have broken the demo.

## Bugs fixed

1. **`.env` not loading.** Your key file is named `env`, not `.env`, and `load_dotenv()` only looks
   for `.env` by default — so `GOOGLE_API_KEY` / `TAVILY_API_KEY` were silently `None` unless you
   happened to be on the original developer's Mac (the code had a hardcoded fallback to
   `/Users/pradyumnareddymagunta/STTP HACK/.env`, which doesn't exist on any other machine).
   `utils.py` now also tries a bare `env` file in the project root. **Keep `env` in the same folder
   as `app.py`** (or rename it to `.env` — either works now).
2. **`confirm_action_node` could return `None`.** If a pending action wasn't `finalize_shortlist`,
   or if the user said "no" for anything other than the shortlist, the function fell through and
   returned `None`, which LangGraph would reject as an invalid state update. Rewrote it with a
   proper `if/elif/else` so every path returns a valid state dict.

## Must Haves — status (all were already solid, kept as-is)

| Requirement | Status |
|---|---|
| Load & parse JD into structured Pydantic fields | ✅ `load_data_node` + `utils.parse_jd` |
| 10-20 resumes | ✅ 15 mock resumes in `data.py` |
| "How many applicants?" with zero LLM calls | ✅ regex-routed straight to `count_applicants_node` |
| "Get me top candidates" → RAG screening with scores | ✅ ChromaDB + `utils.screen_candidates` |
| "Rewrite JD" / "Interview questions" grounded in JD | ✅ `rewrite_jd_node`, `generate_questions_node` |
| "Salary expectations" via live web search, not RAG | ✅ Tavily in `utils.get_salary_benchmark`, with local fallback if no key |
| Human confirms before finalizing a shortlist | ✅ `pending_action` + `confirm_action_node` |
| Terminal only | ✅ `rich`-based CLI in `app.py` |

## Nice to Haves — newly added

- **JD-to-resume mismatch feedback** — after screening, the agent now reports what % of the pool
  actually meets your experience bar and flags it if your JD is unrealistic
  (`utils.compute_jd_mismatch_stats`, surfaced in `screen_candidates_node`).
- **JD improvement suggestions** — right after parsing a JD, the agent flags thin skill lists,
  missing responsibilities, or an unset experience floor (`utils.suggest_jd_improvements`).
- **Candidate comparison** — ask *"Compare John Doe and Jane Smith"* for a markdown side-by-side
  table (new `compare_candidates` intent/node).
- **Skill trend analysis** — ask *"What skills are trending for this role?"* for a **live Tavily
  search** (not RAG) diffed against your JD's required skills, so you see what's missing
  (new `skill_trend` intent/node).
- **Interview scheduling (mock calendar)** — ask *"Schedule an interview with John Doe"*; the agent
  proposes a slot and **waits for your yes/no confirmation** (same human-in-the-loop pattern as
  shortlisting) before booking it to `data/interview_calendar.json`.
- **Actual "sending" of shortlist emails** — emails were previously only drafted and printed; they're
  now also persisted to `data/sent_emails.json` to simulate delivery, since no SMTP/Gmail
  credentials are configured. Swap `utils.send_email_mock` for a real Gmail MCP or SMTP call when
  you're ready to send for real.

## Moon-shots — newly added

- **Resume red-flag detection** — a lightweight, dependency-free heuristic
  (`utils.detect_resume_red_flags`) scans each candidate's original resume text for year-range
  employment gaps and for a mismatch between their stated `experience_years` and their actual
  career timeline. Surfaced automatically whenever you ask for interview questions.
- **Full batch screening report** — *"Give me the full batch report"* returns every screened
  candidate (not just the top 5) plus the JD mismatch stats in one table (new `batch_report` intent/node).

Deliberately **not** implemented (per the brief's "what not to build" guidance): real PDF resume
parsing and a Streamlit/Gradio UI — the terminal agent should prove itself first.

## New commands to try

```
Compare John Doe and Jane Smith
Schedule an interview with Karen White
What skills are trending for this role?
Give me the full batch report
```

## Files changed

- `models.py` — `QueryIntent` now recognizes `compare`, `schedule`, `trend`, `batch_report`.
- `utils.py` — env-loading fix + ~9 new functions (JD suggestions, mismatch stats, comparison,
  skill trends, red-flag detection, mock scheduling, mock email send).
- `agent.py` — router updated for the new intents, 4 new nodes, `confirm_action_node` bug fixed,
  existing nodes enriched with the new feedback.
- `app.py` — dashboard hints and automated demo script (`--test`) extended to exercise every new
  feature end-to-end (now 12 turns instead of 7).
- `data.py`, `pregenerate_cache.py`, `requirement.txt` — unchanged, still work as-is.

## Running it

```bash
pip install -r requirement.txt
python data.py                 # writes data/resumes and data/jds
python pregenerate_cache.py    # pre-caches parsed resume JSON (saves LLM calls/quota)
python app.py                  # interactive terminal chat
python app.py --test           # scripted 12-turn demo covering every feature
```

Make sure `env` (or `.env`) with `GOOGLE_API_KEY` and `TAVILY_API_KEY` sits next to `app.py`.
`TAVILY_API_KEY` is optional — salary lookup and skill trends both fall back to local cached data
if it's missing, they just won't be "live."
