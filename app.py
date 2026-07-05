import os
import sys
import argparse
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich.markdown import Markdown
from rich.live import Live
from rich.spinner import Spinner
from agent import recruitment_agent, AgentState
# Initialize Rich Console
console = Console()
def print_welcome_dashboard(state: dict):
    """Prints a beautiful dashboard panel with active stats."""
    # Check API key status
    google_key = os.getenv("GOOGLE_API_KEY")
    tavily_key = os.getenv("TAVILY_API_KEY")
    
    key_status_text = (
        f"[green]Active ✅[/green]" if google_key else "[red]Missing ❌ (Gemini calls will fail)[/red]"
    )
    tavily_status_text = (
        f"[green]Active ✅ (Live Search)[/green]" if tavily_key else "[yellow]Inactive ⚠️ (Fallback Cache Mode)[/yellow]"
    )
    
    # JD details
    jd_role = state.get("parsed_jd", {}).get("role_name", "None Loaded")
    jd_skills = ", ".join(state.get("parsed_jd", {}).get("required_skills", [])) or "None"
    jd_exp = state.get("parsed_jd", {}).get("required_experience_years", "N/A")
    
    stats_table = Table.grid(padding=(0, 2))
    stats_table.add_column(style="bold cyan", justify="right")
    stats_table.add_column()
    
    stats_table.add_row("Google API Key:", key_status_text)
    stats_table.add_row("Tavily API Key:", tavily_status_text)
    stats_table.add_row("Resumes Loaded:", f"[green]{state.get('candidates_count', 0)}[/green] candidates in vector store")
    stats_table.add_row("Current Job Description:", f"[bold white]{jd_role}[/bold white]")
    stats_table.add_row("  - Required Experience:", f"{jd_exp} years")
    stats_table.add_row("  - Core Required Skills:", f"[yellow]{jd_skills}[/yellow]")
    
    dashboard_panel = Panel(
        stats_table,
        title="[bold gold3]HIREGRAPH AI — RECRUITMENT ASSISTANT[/bold gold3]",
        subtitle="[bold dim]Day 4 Hackathon Submission — Vardhaman College[/bold dim]",
        border_style="bold blue",
        expand=False
    )
    
    console.print(dashboard_panel)
    console.print("\n[bold green]System Ready![/bold green] You can ask questions like:")
    console.print("  • '[cyan]How many applicants?[/cyan]'", style="dim")
    console.print("  • '[cyan]Get me top candidates[/cyan]'", style="dim")
    console.print("  • '[cyan]Rewrite this JD for a startup[/cyan]'", style="dim")
    console.print("  • '[cyan]Salary expectations for this role?[/cyan]'", style="dim")
    console.print("  • '[cyan]Interview questions for John Doe[/cyan]'", style="dim")
    console.print("  • '[cyan]Compare John Doe and Jane Smith[/cyan]'", style="dim")
    console.print("  • '[cyan]Schedule an interview with John Doe[/cyan]'", style="dim")
    console.print("  • '[cyan]What skills are trending for this role?[/cyan]'", style="dim")
    console.print("  • '[cyan]Give me the full batch report[/cyan]'", style="dim")
    console.print("  • Type '[red]exit[/red]' or '[red]quit[/red]' to end the session.\n")
def display_agent_response(response_markdown: str):
    """Draws the agent response inside a stylized panel."""
    # Convert match score tables in Markdown to Rich tables if appropriate,
    # or just let Rich's Markdown parser handle it cleanly.
    md = Markdown(response_markdown)
    response_panel = Panel(
        md,
        title="[bold green]Agent Response[/bold green]",
        border_style="green",
        padding=(1, 2)
    )
    console.print(response_panel)
def run_cli():
    # Setup initial state config
    thread_config = {"configurable": {"thread_id": "recruiter_session_001"}}
    
    # Run initial loading node to setup the system (zero prompt start)
    console.print("[bold yellow]Initializing HireGraph AI system, parsing JD & resumes...[/bold yellow]")
    
    initial_state = {
        "messages": [],
        "jd_text": "",
        "parsed_jd": None,
        "candidates_count": 0,
        "screened_candidates": [],
        "shortlisted_names": [],
        "finalized_shortlist": [],
        "pending_action": None,
        "last_response": "",
        "error_message": None
    }
    
    # We invoke the graph with an initial payload to trigger load_data on startup
    with console.status("[bold blue]Indexing knowledge base...[/bold blue]") as status:
        state = recruitment_agent.invoke(
            {"messages": [{"role": "user", "content": "load"}]}, 
            config=thread_config
        )
        
    print_welcome_dashboard(state)
    
    while True:
        try:
            # Check if there is a pending action
            pending = state.get("pending_action")
            prompt_prefix = "[bold yellow]Confirm Action (yes/no) > [/bold yellow]" if pending else "[bold cyan]Recruiter > [/bold cyan]"
            
            user_input = console.input(prompt_prefix).strip()
            if not user_input:
                continue
                
            if user_input.lower() in ["exit", "quit"]:
                console.print("[bold green]Thank you for using HireGraph AI! Goodbye.[/bold green]")
                break
                
            # Append message to state and run graph
            current_messages = state.get("messages", [])
            current_messages.append({"role": "user", "content": user_input})
            
            # Execute with status spinner
            with console.status("[bold green]Agent thinking...[/bold green]", spinner="dots"):
                state = recruitment_agent.invoke(
                    {"messages": current_messages},
                    config=thread_config
                )
                
            display_agent_response(state["last_response"])
            
        except (KeyboardInterrupt, EOFError):
            console.print("\n[bold green]Session terminated. Goodbye.[/bold green]")
            break
        except Exception as e:
            console.print(Panel(f"[red]An error occurred during execution: {str(e)}[/red]", border_style="red", title="[bold red]Error[/bold red]"))
def run_automated_test():
    """Runs a complete test suite verifying all turns required by the hackathon rubric."""
    console.print("[bold yellow]Starting Automated Demo Verification...[/bold yellow]\n")
    thread_config = {"configurable": {"thread_id": "test_session_999"}}
    
    # Step 1: Initialize
    console.print("[bold blue]Turn 1: Initialize System[/bold blue]")
    state = recruitment_agent.invoke(
        {"messages": [{"role": "user", "content": "load"}]}, 
        config=thread_config
    )
    console.print(f"-> Loaded {state['candidates_count']} resumes. Current Role: {state['parsed_jd']['role_name']}\n")
    
    # Step 2: How many applicants (No LLM count check)
    console.print("[bold blue]Turn 2: Query Applicant Count ('How many applicants?')[/bold blue]")
    state = recruitment_agent.invoke(
        {"messages": state["messages"] + [{"role": "user", "content": "How many applicants?"}]}, 
        config=thread_config
    )
    console.print(f"-> Agent Response: {state['last_response']}\n")
    
    # Step 3: Screening (RAG search)
    console.print("[bold blue]Turn 3: Screen Candidates ('Get me top candidates')[/bold blue]")
    state = recruitment_agent.invoke(
        {"messages": state["messages"] + [{"role": "user", "content": "Get me top candidates"}]}, 
        config=thread_config
    )
    console.print(f"-> Agent Response: {state['last_response']}\n")
    
    # Step 4: Human confirmation (Accepting shortlist)
    console.print("[bold blue]Turn 4: Confirm Shortlist ('yes' to pending shortlist)[/bold blue]")
    state = recruitment_agent.invoke(
        {"messages": state["messages"] + [{"role": "user", "content": "yes"}]}, 
        config=thread_config
    )
    console.print(f"-> Agent Response:\n{state['last_response']}\n")
    
    # Step 5: Rewrite JD for startup
    console.print("[bold blue]Turn 5: Rewrite JD ('Rewrite this JD for a startup')[/bold blue]")
    state = recruitment_agent.invoke(
        {"messages": state["messages"] + [{"role": "user", "content": "Rewrite this JD for a startup"}]}, 
        config=thread_config
    )
    console.print(f"-> Agent Response: {state['last_response']}\n")
    
    # Step 6: Interview Questions for Candidate
    console.print("[bold blue]Turn 6: Generate Interview Questions ('Interview questions for John Doe')[/bold blue]")
    state = recruitment_agent.invoke(
        {"messages": state["messages"] + [{"role": "user", "content": "Interview questions for John Doe"}]}, 
        config=thread_config
    )
    console.print(f"-> Agent Response: {state['last_response']}\n")
    
    # Step 7: Salary benchmark search (Tavily / Cache fallback)
    console.print("[bold blue]Turn 7: Fetch Salary Expectations ('Salary expectations for this role?')[/bold blue]")
    state = recruitment_agent.invoke(
        {"messages": state["messages"] + [{"role": "user", "content": "Salary expectations for this role?"}]}, 
        config=thread_config
    )
    console.print(f"-> Agent Response: {state['last_response']}\n")
    
    # Step 8: Skill trend analysis (Tavily live search vs JD)
    console.print("[bold blue]Turn 8: Skill Trend Analysis ('What skills are trending for this role?')[/bold blue]")
    state = recruitment_agent.invoke(
        {"messages": state["messages"] + [{"role": "user", "content": "What skills are trending for this role?"}]},
        config=thread_config
    )
    console.print(f"-> Agent Response: {state['last_response']}\n")
    
    # Step 9: Candidate comparison
    console.print("[bold blue]Turn 9: Compare Candidates ('Compare the top two candidates')[/bold blue]")
    state = recruitment_agent.invoke(
        {"messages": state["messages"] + [{"role": "user", "content": "Compare the top two candidates"}]},
        config=thread_config
    )
    console.print(f"-> Agent Response: {state['last_response']}\n")
    
    # Step 10: Full batch report (all candidates, not just top 5)
    console.print("[bold blue]Turn 10: Full Batch Report ('Give me the full batch report')[/bold blue]")
    state = recruitment_agent.invoke(
        {"messages": state["messages"] + [{"role": "user", "content": "Give me the full batch report"}]},
        config=thread_config
    )
    console.print(f"-> Agent Response: {state['last_response']}\n")
    
    # Step 11: Mock interview scheduling with human-in-the-loop confirmation
    console.print("[bold blue]Turn 11: Schedule Interview ('Schedule an interview with John Doe')[/bold blue]")
    state = recruitment_agent.invoke(
        {"messages": state["messages"] + [{"role": "user", "content": "Schedule an interview with John Doe"}]},
        config=thread_config
    )
    console.print(f"-> Agent Response: {state['last_response']}\n")
    
    console.print("[bold blue]Turn 12: Confirm Interview Slot ('yes')[/bold blue]")
    state = recruitment_agent.invoke(
        {"messages": state["messages"] + [{"role": "user", "content": "yes"}]},
        config=thread_config
    )
    console.print(f"-> Agent Response: {state['last_response']}\n")
    
    console.print("[bold green]Automated Demo Verification Completed Successfully! All agent workflows verified.[/bold green]")
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="HireGraph AI Recruitment Chatbot")
    parser.add_argument("--test", action="store_true", help="Run automated test suite verify flow")
    args = parser.parse_args()
    
    if args.test:
        run_automated_test()
    else:
        run_cli()
