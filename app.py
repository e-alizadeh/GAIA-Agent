import os
import gradio as gr
import requests
import inspect
import pandas as pd

# (Keep Constants and BasicAgent class as is)
# --- Constants ---
DEFAULT_API_URL = "https://jofthomas-unit4-scoring.hf.space/"

# --- Basic Agent Definition ---
class BasicAgent:
    # ... (keep agent code as is) ...
    def __init__(self):
        print("BasicAgent initialized.")
    def __call__(self, question: str) -> str:
        print(f"Agent received question (first 50 chars): {question[:50]}...")
        fixed_answer = "This is a default answer."
        print(f"Agent returning fixed answer: {fixed_answer}")
        return fixed_answer
    def __repr__(self) -> str:
        imports = ["import inspect\n"]
        try:
            class_source = inspect.getsource(BasicAgent)
            full_source = "\n".join(imports) + "\n" + class_source
            return full_source
        except Exception as e:
            print(f"Error getting source code via inspect: {e}")
            return f"# Could not get source via inspect: {e}"

# --- Gradio UI and Logic ---
def get_current_script_content() -> str:
    # ... (keep function as is) ...
    try:
        script_path = os.path.abspath(__file__)
        print(f"Reading script content from: {script_path}")
        with open(script_path, 'r', encoding='utf-8') as f:
            return f.read()
    except NameError:
        print("Warning: __file__ is not defined. Cannot read script content this way.")
        return "# Agent code unavailable: __file__ not defined"
    except FileNotFoundError:
        print(f"Warning: Script file '{script_path}' not found.")
        return f"# Agent code unavailable: Script file not found at {script_path}"
    except Exception as e:
        print(f"Error reading script file '{script_path}': {e}")
        return f"# Agent code unavailable: Error reading script file: {e}"


def run_and_submit_all( profile: gr.OAuthProfile | None):
    """
    Fetches all questions, runs the BasicAgent on them, submits all answers,
    and displays the results.
    """
    # --- Determine HF Space Runtime URL and Repo URL ---
    space_host = os.getenv("SPACE_HOST")
    space_id = os.getenv("SPACE_ID") # Get the SPACE_ID

    hf_runtime_url = "Runtime: Locally or unknown environment (SPACE_HOST not found)"
    hf_repo_url = "HF Repo URL: Unknown (SPACE_ID not found)"
    hf_repo_tree_url = "HF Repo Tree URL: Unknown (SPACE_ID not found)"

    if space_host:
         hf_runtime_url = f"Runtime URL: https://{space_host}.hf.space"

    if space_id: # Construct URLs using SPACE_ID
        hf_repo_url = f"HF Repo URL: https://huggingface.co/spaces/{space_id}"
        hf_repo_tree_url = f"HF Repo Tree URL: https://huggingface.co/spaces/{space_id}/tree/main"

    # Print runtime and repo info at the start
    print("\n" + "="*60)
    print("Executing run_and_submit_all function...")
    print(hf_runtime_url) # Print the runtime URL (from SPACE_HOST)
    print(hf_repo_url)    # Print the base repo URL (from SPACE_ID)
    print(hf_repo_tree_url) # Print the repo tree URL (from SPACE_ID)
    # --- End Environment Info ---

    if profile:
        username= f"{profile.username}"
        print(f"User logged in: {username}")
    else:
        print("User not logged in.")
        print("="*60 + "\n")
        return "Please Login to Hugging Face with the button.", None

    print("="*60 + "\n")

    # ... (rest of the function remains the same) ...
    api_url = DEFAULT_API_URL
    questions_url = f"{api_url}/questions"
    submit_url = f"{api_url}/submit"

    # 1. Instantiate Agent
    try:
        agent = BasicAgent()
    except Exception as e:
        print(f"Error instantiating agent: {e}")
        return f"Error initializing agent: {e}", None
    agent_code = get_current_script_content()
    if agent_code.startswith("# Agent code unavailable"):
        print("Warning: Using potentially incomplete agent code due to reading error.")

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

    # 3. Run Agent
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
        "Please clone this space, then modify the code to define your agent's logic within the `BasicAgent` class. "
        "Log in to your Hugging Face account using the button below. This uses your HF username for submission. "
        "Click 'Run Evaluation & Submit All Answers' to fetch questions, run your agent, submit answers, and see the score."
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