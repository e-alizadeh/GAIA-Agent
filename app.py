import os
import re
from typing import Literal, TypedDict, get_args

import gradio as gr
import pandas as pd
import requests
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph

from tools import (
    analyze_excel_file,
    calculator,
    image_describe,
    run_py,
    transcribe_via_whisper,
    web_multi_search,
    wiki_search,
    youtube_transcript,
)

# --------------------------------------------------------------------------- #
#                              CONFIGURATION                                  #
# --------------------------------------------------------------------------- #
DEFAULT_API_URL: str = "https://agents-course-unit4-scoring.hf.space"
MODEL_NAME: str = "o4-mini"  # "gpt-4.1-mini"
TEMPERATURE: float = 0.1

_SYSTEM_PROMPT = """You are a precise research assistant. Return ONLY the literal answer - no preamble.
If the question asks for a *first name*, output the first given name only.
If the answer is numeric, output digits only (no commas, units, or words).
"""

# --------------------------------------------------------------------------- #
#                           QUESTION  CLASSIFIER                               #
# --------------------------------------------------------------------------- #

_LABELS = Literal["math", "youtube", "image", "code", "excel", "audio", "general"]

_CLASSIFY_PROMPT = """You are a router that labels the user question with exactly one of the following categories:
{labels}.

User question:
{question}

Label:
"""


# --------------------------------------------------------------------------- #
# -------------------------------  AGENT STATE  ----------------------------- #
# --------------------------------------------------------------------------- #
class AgentState(TypedDict):
    question: str
    label: str
    context: str
    answer: str
    confidence: float
    task_id: str | None = None


# --------------------------------------------------------------------------- #
#                         NODES  (LangGraph  functions)                        #
# --------------------------------------------------------------------------- #

_llm_router = ChatOpenAI(model=MODEL_NAME)
_llm_answer = ChatOpenAI(model=MODEL_NAME)


def classify(state: AgentState) -> AgentState:  # noqa: D401
    """Label the task so we know which toolchain to invoke."""
    question = state["question"]

    label_values = set(get_args(_LABELS))  # -> ("math", "youtube", ...)
    parsed_labels = ", ".join(repr(v) for v in label_values)
    resp = (
        _llm_router.invoke(
            _CLASSIFY_PROMPT.format(question=question, labels=parsed_labels)
        )
        .content.strip()
        .lower()
    )
    state["label"] = resp if resp in label_values else "general"
    return state


def gather_context(state: AgentState) -> AgentState:
    question, label, task_id = state["question"], state["label"], state["task_id"]

    matched_pattern = r"https?://\S+"
    matched_obj = re.search(matched_pattern, question)

    # ---- attachment detection ------------------------------------------------
    if task_id:
        file_url = f"{DEFAULT_API_URL}/files/{task_id}"
        head = requests.head(file_url, timeout=10)
        ctype = head.headers.get("content-type", "")

        print(f"[DEBUG] attachment type={ctype} | url={file_url}")
        if "python" in ctype or file_url.endswith(".py"):
            code = requests.get(file_url, timeout=10).text
            state["answer"] = run_py.invoke({"code": code})
            state["label"] = "code"
            return state
        if "excel" in ctype or file_url.endswith((".xlsx", ".csv")):
            blob = requests.get(file_url, timeout=10).content
            state["context"] = analyze_excel_file.invoke(
                {"xls_bytes": blob, "question": question}
            )
            state["label"] = "excel"
            return state
        if "audio" in ctype or file_url.endswith(".mp3"):
            blob = requests.get(file_url, timeout=10).content
            state["context"] = transcribe_via_whisper.invoke({"mp3_bytes": blob})
            state["label"] = "audio"
            return state

    if label == "math":
        print("[TOOL] calculator")
        expr = re.sub(r"\s+", "", question)
        state["context"] = calculator.invoke({"expression": expr})
    elif label == "youtube" and matched_obj:
        print("[TOOL] youtube_transcript")
        if matched_obj:
            url = matched_obj[0]
            state["context"] = youtube_transcript.invoke({"url": url})
    elif label == "image" and matched_obj:
        print("[TOOL] image")
        if matched_obj:
            url = matched_obj[0]
            state["context"] = image_describe.invoke({"image_url": url})
    else:  # general
        print("[TOOL] general")
        search_json = web_multi_search.invoke({"query": question})
        wiki_text = wiki_search.invoke({"query": question})
        state["context"] = f"{search_json}\n\n{wiki_text}"

    return state


def generate_answer(state: AgentState) -> AgentState:
    # Skip LLM for deterministic labels
    if state["label"] in {"math", "code", "excel"}:
        state["confidence"] = 0.9
        return state

    prompt = [
        SystemMessage(content=_SYSTEM_PROMPT),
        HumanMessage(
            content=f"Question: {state['question']}\n\nContext:\n{state['context']}\n\nAnswer:"
        ),
    ]
    raw = _llm_answer.invoke(prompt).content.strip()
    state["answer"] = raw
    state["confidence"] = 0.5
    return state


def validate(state: AgentState) -> AgentState:
    """Simple format + confidence gate."""
    txt = re.sub(r"^(final answer:?\s*)", "", state["answer"], flags=re.I).strip()

    # If question demands a single token (first name / one word), enforce it
    if any(kw in state["question"].lower() for kw in ["first name", "single word"]):
        txt = txt.split(" ")[0]

    txt = txt.rstrip(".")
    if not txt or len(txt.split()) > 6 or state["confidence"] < 0.2:
        txt = "I don’t know"

    state["answer"] = txt
    return state


# --------------------------------------------------------------------------- #
#                              BUILD  THE  GRAPH                              #
# --------------------------------------------------------------------------- #


def build_graph() -> StateGraph:
    g = StateGraph(AgentState)
    g.set_entry_point("classify")

    g.add_node("classify", classify)
    g.add_node("gather", gather_context)
    g.add_node("generate", generate_answer)
    g.add_node("validate", validate)

    g.add_edge("classify", "gather")
    g.add_edge("gather", "generate")
    g.add_edge("generate", "validate")
    g.add_edge("validate", END)

    return g.compile()


# --------------------------------------------------------------------------- #
# -------------------------------  GAIA AGENT  ------------------------------ #
# --------------------------------------------------------------------------- #
class GAIAAgent:
    """Callable wrapper used by run_and_submit_all."""

    def __init__(self) -> None:
        self.graph = build_graph()

    def __call__(self, question: str, task_id: str | None = None) -> str:
        state: AgentState = {
            "question": question,
            "label": "general",
            "context": "",
            "answer": "",
            "confidence": 0.0,
            "task_id": task_id,
        }
        final = self.graph.invoke(state)

        # ── Debug trace ───────────────────────────────────────────────
        route = final["label"]
        llm_used = route != "math"  # math path skips the generation LLM
        print(f"[DEBUG] route='{route}' | LLM_used={llm_used}")
        # ─────────────────────────────────────────────────────────────

        return final["answer"]


def run_and_submit_all(
    profile: gr.OAuthProfile | None,
) -> tuple[str, pd.DataFrame | None]:
    """
    Fetches all questions, runs the BasicAgent on them, submits all answers,
    and displays the results.
    """
    # --- Determine HF Space Runtime URL and Repo URL ---
    space_id = os.getenv("SPACE_ID")  # Get the SPACE_ID for sending link to the code

    if profile:
        username = f"{profile.username}"
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
            submitted_answer = agent(question=question_text, task_id=task_id)
            answers_payload.append(
                {"task_id": task_id, "submitted_answer": submitted_answer}
            )
            results_log.append(
                {
                    "Task ID": task_id,
                    "Question": question_text,
                    "Submitted Answer": submitted_answer,
                }
            )
        except Exception as e:
            print(f"Error running agent on task {task_id}: {e}")
            results_log.append(
                {
                    "Task ID": task_id,
                    "Question": question_text,
                    "Submitted Answer": f"AGENT ERROR: {e}",
                }
            )

    if not answers_payload:
        print("Agent did not produce any answers to submit.")
        return "Agent did not produce any answers to submit.", pd.DataFrame(results_log)

    # 4. Prepare Submission
    submission_data = {
        "username": username.strip(),
        "agent_code": agent_code,
        "answers": answers_payload,
    }
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

    status_output = gr.Textbox(
        label="Run Status / Submission Result", lines=5, interactive=False
    )
    # Removed max_rows=10 from DataFrame constructor
    results_table = gr.DataFrame(label="Questions and Agent Answers", wrap=True)

    run_button.click(fn=run_and_submit_all, outputs=[status_output, results_table])


if __name__ == "__main__":
    print("\n" + "-" * 30 + " App Starting " + "-" * 30)
    # Check for SPACE_HOST and SPACE_ID at startup for information
    space_host_startup = os.getenv("SPACE_HOST")
    space_id_startup = os.getenv("SPACE_ID")  # Get SPACE_ID at startup

    if space_host_startup:
        print(f"✅ SPACE_HOST found: {space_host_startup}")
        print(f"   Runtime URL should be: https://{space_host_startup}.hf.space")
    else:
        print("ℹ️  SPACE_HOST environment variable not found (running locally?).")

    if space_id_startup:  # Print repo URLs if SPACE_ID is found
        print(f"✅ SPACE_ID found: {space_id_startup}")
        print(f"   Repo URL: https://huggingface.co/spaces/{space_id_startup}")
        print(
            f"   Repo Tree URL: https://huggingface.co/spaces/{space_id_startup}/tree/main"
        )
    else:
        print(
            "ℹ️  SPACE_ID environment variable not found (running locally?). Repo URL cannot be determined."
        )

    print("-" * (60 + len(" App Starting ")) + "\n")

    print("Launching Gradio Interface for Basic Agent Evaluation...")
    demo.launch(debug=True, share=False)


## For Local testing
# if __name__ == "__main__":
#     agent = GAIAAgent()
#     while True:
#         try:
#             q = input("\nEnter question (or blank to quit): ")
#         except KeyboardInterrupt:
#             break
#         if not q.strip():
#             break
#         print("Answer:", agent(q))
