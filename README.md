# HireGraph AI

An AI-powered Recruitment Chatbot built using LangGraph, LangChain, ChromaDB, Google Gemini, and Tavily. HireGraph AI is an intelligent recruitment assistant that helps recruiters screen resumes against a job description, generate grounded interview questions, benchmark salaries, and manage a hiring shortlist — all through Retrieval-Augmented Generation (RAG), vector search, and live web search for market data.

The chatbot operates entirely through a terminal interface and uses a LangGraph workflow to orchestrate JD parsing, candidate screening, interview prep, comparison, scheduling, and human-in-the-loop shortlist approval.

## Team

**Team Name:** Quantum Crew

Members:

 M Pradyumna Reddy
 Pindi Rupesh


## Features

- Job description parsing into structured fields
- Resume screening using Hybrid RAG
- ChromaDB vector search
- Google Gemini embeddings + LLM
- Tavily live web search integration (salary + skill trends)
- Zero-LLM-cost regex routing for simple queries (e.g. "how many applicants?")
- Human-in-the-loop shortlist confirmation
- Interview question generation grounded in JD + resume gaps
- Resume red-flag / integrity detection
- Candidate comparison
- Salary benchmarking
- Interview scheduling (mock calendar)
- Full batch screening report
- Rich-based terminal interface

## Tech Stack

| Technology              | Purpose                     |
| ------------------------ | ---------------------------- |
| Python                   | Backend                      |
| LangChain                | RAG Pipeline                 |
| LangGraph                | Workflow orchestration       |
| ChromaDB                 | Vector Database              |
| Google Gemini            | LLM + Embeddings             |
| Tavily                   | Live web search (salary/trends) |
| Pydantic                 | Structured data models       |
| Rich                     | Terminal UI                  |
| python-dotenv            | Environment variables        |

## Project Architecture

```
HireGraph AI
        │
        ▼
    User Query
        │
        ▼
      Router
        │
 ┌──────┼───────────────┬───────────────┐
 │      │               │               │
Load   Count         Screen          Chat
JD    Applicants   Candidates
                        │
                        ▼
                 ┌──────┴──────┐
                 │             │
            Interview      Compare /
            Questions      Salary /
                            Trend /
                            Schedule
                        │
                        ▼
                Confirm Action
                        │
                        ▼
                Shortlist / Email
```

## Folder Structure

```
Project/
│
├── data/
│   ├── jds/
│   │   └── ai_engineer_jd.txt
│   ├── resumes/
│   ├── parsed_resumes/
│   └── chroma_db/            (generated, gitignored)
│
├── agent.py                  # LangGraph state, router, nodes
├── app.py                    # Rich-based CLI entry point
├── models.py                 # Pydantic schemas
├── utils.py                  # LLM calls, RAG, scoring, Tavily lookups
├── data.py                   # Generates mock resumes/JD
├── pregenerate_cache.py      # Pre-caches parsed resume JSON
├── requirement.txt
└── README.md
```

## Installation

```bash
git clone <repository>

cd Project

pip install -r requirement.txt
```

## Environment Variables

Create a `.env` file in the project root:

```
GOOGLE_API_KEY=
TAVILY_API_KEY=
```

`TAVILY_API_KEY` is optional — salary lookup and skill trend analysis fall back to cached local data if it's missing, they just won't be "live."

## Running the Project

```bash
python data.py                 # writes data/resumes and data/jds
python pregenerate_cache.py    # pre-caches parsed resume JSON (saves LLM calls/quota)
python app.py                  # interactive terminal chat
python app.py --test           # scripted end-to-end demo
```

## Workflow

```
Start
   │
   ▼
Load Job Description
   │
   ▼
Parse & Index Resumes into ChromaDB
   │
   ▼
Receive User Query
   │
   ▼
Router
   │
   ▼
Hybrid RAG Retrieval
   │
   ▼
Candidate Ranking
   │
   ▼
Interview Prep / Comparison / Salary / Trend
   │
   ▼
Shortlist Confirmation (Human-in-the-loop)
   │
   ▼
Email Draft & Send (simulated)
```

## Example Conversation

User:
Get me top candidates

↓

RAG Candidate Screening Results

1. John Doe — Match Score: 91%
2. Jane Smith — Match Score: 86%
3. Alex Kumar — Match Score: 82%

↓

User:
Interview questions for John Doe

↓

Customized interview questions
Resume integrity check

↓

User:
Finalize this shortlist

↓

YES

↓

[SIMULATED] Shortlist emails sent

## Future Improvements

- Real PDF resume parsing
- Streamlit/Gradio web interface
- Gmail API integration for real email sending
- Multi-role batch hiring pipelines
- Applicant tracking system (ATS) integration
- Authentication for recruiter accounts
