import ast
from functools import lru_cache
import json
import operator
import os
import re
from typing import Annotated, TypedDict
import gradio as gr
from langchain_openai import ChatOpenAI
import requests
import pandas as pd
from langchain_community.utilities import DuckDuckGoSearchAPIWrapper
from langchain.agents import tool
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolExecutor
from langgraph.graph import StateGraph, END

# --- Constants ---
DEFAULT_API_URL = "https://agents-course-unit4-scoring.hf.space"

# ----- THIS IS WERE YOU CAN BUILD WHAT YOU WANT ------
# --------------------------------------------------------------------------- #
# -----------------------------  SAFE CALCULATOR  --------------------------- #
# --------------------------------------------------------------------------- #
_ALLOWED_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
}

def _safe_eval(node: ast.AST) -> float:
    if isinstance(node, ast.Num):  # literal number
        return node.n
    if isinstance(node, ast.UnaryOp) and type(node.op) in _ALLOWED_OPS:
        return _ALLOWED_OPS[type(node.op)](_safe_eval(node.operand))
    if isinstance(node, ast.BinOp) and type(node.op) in _ALLOWED_OPS:
        return _ALLOWED_OPS[type(node.op)](
            _safe_eval(node.left), _safe_eval(node.right)
        )
    raise ValueError("Unsafe or unsupported expression")

@tool
def calculator(expression: str) -> str:
    """Calculate mathematical expressions safely."""
    try:
        tree = ast.parse(expression, mode="eval")
        return str(_safe_eval(tree.body))
    except Exception as exc:
        return f"calc_error:{exc}"

# --------------------------------------------------------------------------- #
# -----------------------------     WEB SEARCH    --------------------------- #
# --------------------------------------------------------------------------- #
@lru_cache(maxsize=128)
def _search_duckduckgo(query: str, k: int = 5) -> list[dict]:
    """Returns the top-k DuckDuckGo results as a list of {title, snippet, link}. Caches identical queries."""

    wrapper = DuckDuckGoSearchAPIWrapper(max_results=k)
    raw = wrapper.results(query)
    cleaned = []
    for hit in raw[:k]:
        cleaned.append(
            {
                "title": hit.get("title", "")[:120],
                "snippet": hit.get("snippet", "")[:200],
                "link": hit.get("link", "")[:200],
            }
        )
    return cleaned

@tool
def web_search(query: str) -> str:
    """DuckDuckGo search. Returns compact JSON (max 5 hits)."""
    try:
        return json.dumps(_search_duckduckgo(query), ensure_ascii=False)
    except Exception as exc:
        return f"search_error:{exc}"

# --------------------------------------------------------------------------- #
# -------------------------------  AGENT STATE  ----------------------------- #
# --------------------------------------------------------------------------- #
class AgentState(TypedDict):
    msg: Annotated[list[BaseMessage], add_messages]
    question: str
    answer: str
    search_results: str 
    reasoning_steps: list[str]
    tools_used: list[str]

# --------------------------------------------------------------------------- #
# -------------------------------  GAIA AGENT  ------------------------------ #
# --------------------------------------------------------------------------- #
class GAIAAgent:
    """
    LangGraph-powered agent targeting GAIA Level-1 tasks.
    Key design points:
      - Compact, cached web results -> lower token cost, less noise.
      - Safe calculator (no eval foot-gun).
      - Answer canonicalisation to hit GAIA's exact-match scoring.
    """

    SYSTEM_PROMPT = (
        "You are an expert question-answering agent. "
        "Return ONLY the final answer—no rationale, no extra words."
    )

    def __init__(self):
        try:
            self.llm = ChatOpenAI(
                model="gpt-3.5-turbo", 
                temperature=0.1,
                api_key=os.getenv("OPENAI_API_KEY")
            )
        except Exception as e:
            print(f"Warning: Could not initialize OpenAI model: {e}")
            self.llm = None

        # Define tools & executor
        self.tools = [web_search, calculator]
        self.tool_executor = ToolExecutor(self.tools)

        self.graph = self._build_graph()

    def _build_graph(self) -> StateGraph:
        """Build the LangGraph workflow."""
        workflow = StateGraph(AgentState)

        # Add nodes
        workflow.add_node("analyze_question", self._analyze_question)
        workflow.add_node("search", self._search_info)
        workflow.add_node("process", self._process_info)
        workflow.add_node("generate_answer", self._generate_answer)
        workflow.add_node("verify_answer", self._verify_answer)

        # Add edges
        workflow.set_entry_point("analyze_question")
        workflow.add_edge("analyze_question", "search")
        workflow.add_edge("search", "process")
        workflow.add_edge("process", "generate_answer")
        workflow.add_edge("generate_answer", "verify")
        workflow.add_edge("verify", END)

        return workflow.compile()

    # ------------------ NODE IMPLEMENTATIONS ------------------ #
    def _extract_search_terms(self, question: str) -> str:
        stops = {
            "what", "who", "where", "when", "how", "why",
            "is", "are", "was", "were", "the", "and", "or",
        }
        tokens = re.findall(r"[A-Za-z0-9]+", question.lower())
        key = [tok for tok in tokens if tok not in stops][:6]
        return " ".join(key)

    def _analyse_question(self, state: AgentState) -> AgentState:
        q = state["question"]
        state["reasoning_steps"] = [f"analyse:{q[:60]}…"]
        return state
    def _search_information(self, state: AgentState) -> AgentState:
        query = self._extract_search_terms(state["question"])
        state["reasoning_steps"].append(f"search:{query}")
        results_json = web_search.invoke({"query": query})
        state["search_results"] = results_json
        state["tools_used"].append("web_search")
        return state

    def _generate_answer(self, state: AgentState) -> AgentState:
        prompt = [
            SystemMessage(content=self.SYSTEM_PROMPT),
            HumanMessage(
                content=(
                    f"Question: {state['question']}\n"
                    f"Search Results (JSON): {state['search_results']}\n"
                    f"Answer:"
                )
            ),
        ]
        rsp = self.llm(prompt)
        state["answer"] = rsp.content.strip()
        state["reasoning_steps"].append("synth")
        return state

    
    def _fallback_answer(self, question: str, search_results: str) -> str:
        """Fallback answer generation without LLM using rule-based reasoning."""
        if "yes" in question.lower() or "no" in question.lower():
            return "Yes" if "yes" in search_results.lower() else "No"

        # Extract numbers for numeric questions
        if any(word in question.lower() for word in ["how many", "count", "number"]):
            numbers = re.findall(r'\d+', search_results)
            if numbers:
                return numbers[0]
            
        # Extract years for date questions
        if "when" in question.lower() or "year" in question.lower():
            years = re.findall(r'\b(19|20)\d{2}\b', search_results)
            if years:
                return years[0]

        # Default: extract first sentence from search results
        sentences = search_results.split('.')
        if sentences:
            return sentences[0].strip()
        
        return "Unable to determine answer from available information."
    

    def _verify_answer(self, state: AgentState) -> AgentState:
        ans = state["answer"].strip()

        # Canonicalise numbers (remove commas) / lowercase yes|no
        if re.fullmatch(r"[0-9][0-9,\.]*", ans):
            ans = ans.replace(",", "")
        if ans.lower() in {"yes", "no"}:
            ans = ans.capitalize()

        if not ans:
            ans = "No answer found"

        state["answer"] = ans
        state["reasoning_steps"].append("Answer verification completed.")
        return state
    
    def __call__(self, question: str) -> str:
        """Main agent call method."""
        print(f"GAIA Agent processing question: {question[:100]}...")
        
        try:
            initial_state: AgentState = {
                "messages": [],
                "question": question,
                "answer": "",
                "search_results": "",
                "reasoning_steps": [],
                "tools_used": []
            }
            
            # Run the graph
            final_state = self.graph.invoke(initial_state)
            
            answer = final_state["answer"]
            print(f"Agent reasoning: {' -> '.join(final_state['reasoning_steps'])}")
            print(f"Tools used: {final_state['tools_used']}")
            print(f"Final answer: {answer}")
            
            return answer
            
        except Exception as e:
            print(f"Error in agent processing: {e}")
            return f"Error processing question: {str(e)}"


def run_and_submit_all(profile: gr.OAuthProfile | None):
    """
    Fetches all questions, runs the BasicAgent on them, submits all answers,
    and displays the results.
    """
    # --- Determine HF Space Runtime URL and Repo URL ---
    space_id = os.getenv("SPACE_ID") # Get the SPACE_ID for sending link to the code

    if profile:
        username= f"{profile.username}"
        print(f"User logged in: {username}")
    else:
        print("User not logged in.")
        return "Please Login to Hugging Face with the button.", None

    api_url = DEFAULT_API_URL
    questions_url = f"{api_url}/questions"
    submit_url = f"{api_url}/submit"

    # 1. Instantiate Agent ( modify this part to create your agent)
    try:
        agent = GAIAAgent()
        print("GAIA Agent initialized successfully")
    except Exception as e:
        print(f"Error instantiating agent: {e}")
        return f"Error initializing agent: {e}", None
    # In the case of an app running as a hugging Face space, this link points toward your codebase ( usefull for others so please keep it public)
    agent_code = f"https://huggingface.co/spaces/{space_id}/tree/main"
    print(agent_code)

    # 2. Fetch Questions
    print(f"Fetching questions from: {questions_url}")
    try:
        response = requests.get(questions_url, timeout=15)
        response.raise_for_status()
        questions_data = response.json()
        if not questions_data:
             print("Fetched questions list is empty.")
             return "Fetched questions list is empty or invalid format.", None
        print(f"Fetched {len(questions_data)} questions.")
    except requests.exceptions.RequestException as e:
        print(f"Error fetching questions: {e}")
        return f"Error fetching questions: {e}", None
    except requests.exceptions.JSONDecodeError as e:
         print(f"Error decoding JSON response from questions endpoint: {e}")
         print(f"Response text: {response.text[:500]}")
         return f"Error decoding server response for questions: {e}", None
    except Exception as e:
        print(f"An unexpected error occurred fetching questions: {e}")
        return f"An unexpected error occurred fetching questions: {e}", None

    # 3. Run your Agent
    results_log = []
    answers_payload = []
    print(f"Running agent on {len(questions_data)} questions...")
    for item in questions_data:
        task_id = item.get("task_id")
        question_text = item.get("question")
        if not task_id or question_text is None:
            print(f"Skipping item with missing task_id or question: {item}")
            continue
        try:
            submitted_answer = agent(question_text)
            answers_payload.append({"task_id": task_id, "submitted_answer": submitted_answer})
            results_log.append({"Task ID": task_id, "Question": question_text, "Submitted Answer": submitted_answer})
        except Exception as e:
             print(f"Error running agent on task {task_id}: {e}")
             results_log.append({"Task ID": task_id, "Question": question_text, "Submitted Answer": f"AGENT ERROR: {e}"})

    if not answers_payload:
        print("Agent did not produce any answers to submit.")
        return "Agent did not produce any answers to submit.", pd.DataFrame(results_log)

    # 4. Prepare Submission 
    submission_data = {"username": username.strip(), "agent_code": agent_code, "answers": answers_payload}
    status_update = f"Agent finished. Submitting {len(answers_payload)} answers for user '{username}'..."
    print(status_update)

    # 5. Submit
    print(f"Submitting {len(answers_payload)} answers to: {submit_url}")
    try:
        response = requests.post(submit_url, json=submission_data, timeout=60)
        response.raise_for_status()
        result_data = response.json()
        final_status = (
            f"Submission Successful!\n"
            f"User: {result_data.get('username')}\n"
            f"Overall Score: {result_data.get('score', 'N/A')}% "
            f"({result_data.get('correct_count', '?')}/{result_data.get('total_attempted', '?')} correct)\n"
            f"Message: {result_data.get('message', 'No message received.')}"
        )
        print("Submission successful.")
        results_df = pd.DataFrame(results_log)
        return final_status, results_df
    except requests.exceptions.HTTPError as e:
        error_detail = f"Server responded with status {e.response.status_code}."
        try:
            error_json = e.response.json()
            error_detail += f" Detail: {error_json.get('detail', e.response.text)}"
        except requests.exceptions.JSONDecodeError:
            error_detail += f" Response: {e.response.text[:500]}"
        status_message = f"Submission Failed: {error_detail}"
        print(status_message)
        results_df = pd.DataFrame(results_log)
        return status_message, results_df
    except requests.exceptions.Timeout:
        status_message = "Submission Failed: The request timed out."
        print(status_message)
        results_df = pd.DataFrame(results_log)
        return status_message, results_df
    except requests.exceptions.RequestException as e:
        status_message = f"Submission Failed: Network error - {e}"
        print(status_message)
        results_df = pd.DataFrame(results_log)
        return status_message, results_df
    except Exception as e:
        status_message = f"An unexpected error occurred during submission: {e}"
        print(status_message)
        results_df = pd.DataFrame(results_log)
        return status_message, results_df


# --- Build Gradio Interface using Blocks ---
with gr.Blocks() as demo:
    gr.Markdown("# Basic Agent Evaluation Runner")
    gr.Markdown(
        """
        **Instructions:**

        1.  Please clone this space, then modify the code to define your agent's logic, the tools, the necessary packages, etc ...
        2.  Log in to your Hugging Face account using the button below. This uses your HF username for submission.
        3.  Click 'Run Evaluation & Submit All Answers' to fetch questions, run your agent, submit answers, and see the score.

        ---
        **Disclaimers:**
        Once clicking on the "submit button, it can take quite some time ( this is the time for the agent to go through all the questions).
        This space provides a basic setup and is intentionally sub-optimal to encourage you to develop your own, more robust solution. For instance for the delay process of the submit button, a solution could be to cache the answers and submit in a seperate action or even to answer the questions in async.
        """
    )

    gr.LoginButton()

    run_button = gr.Button("Run Evaluation & Submit All Answers")

    status_output = gr.Textbox(label="Run Status / Submission Result", lines=5, interactive=False)
    # Removed max_rows=10 from DataFrame constructor
    results_table = gr.DataFrame(label="Questions and Agent Answers", wrap=True)

    run_button.click(
        fn=run_and_submit_all,
        outputs=[status_output, results_table]
    )

if __name__ == "__main__":
    print("\n" + "-"*30 + " App Starting " + "-"*30)
    # Check for SPACE_HOST and SPACE_ID at startup for information
    space_host_startup = os.getenv("SPACE_HOST")
    space_id_startup = os.getenv("SPACE_ID") # Get SPACE_ID at startup

    if space_host_startup:
        print(f"✅ SPACE_HOST found: {space_host_startup}")
        print(f"   Runtime URL should be: https://{space_host_startup}.hf.space")
    else:
        print("ℹ️  SPACE_HOST environment variable not found (running locally?).")

    if space_id_startup: # Print repo URLs if SPACE_ID is found
        print(f"✅ SPACE_ID found: {space_id_startup}")
        print(f"   Repo URL: https://huggingface.co/spaces/{space_id_startup}")
        print(f"   Repo Tree URL: https://huggingface.co/spaces/{space_id_startup}/tree/main")
    else:
        print("ℹ️  SPACE_ID environment variable not found (running locally?). Repo URL cannot be determined.")

    print("-"*(60 + len(" App Starting ")) + "\n")

    print("Launching Gradio Interface for Basic Agent Evaluation...")
    demo.launch(debug=True, share=False)