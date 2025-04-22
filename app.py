import os
import gradio as gr
import requests
import inspect # To get source code for __repr__
import pandas as pd # For displaying results in a table

# --- Constants ---
DEFAULT_API_URL = "http://127.0.0.1:8000" # Default URL for your FastAPI app

# --- Basic Agent Definition ---

class BasicAgent:
    """
    A very simple agent placeholder.
    It just returns a fixed string for any question.
    """
    def __init__(self):
        print("BasicAgent initialized.")
        # Add any setup if needed

    def __call__(self, question: str) -> str:
        """
        The agent's logic to answer a question.
        This basic version ignores the question content.
        """
        print(f"Agent received question (first 50 chars): {question[:50]}...")
        # Replace this with actual logic if you were building a real agent
        fixed_answer = "This is a default answer."
        print(f"Agent returning fixed answer: {fixed_answer}")
        return fixed_answer

    def __repr__(self) -> str:
        """
        Return the source code required to reconstruct this agent.
        """
        imports = [
            "import inspect\n" # May not be strictly needed by the agent logic itself
        ]
        class_source = inspect.getsource(BasicAgent)
        full_source = "\n".join(imports) + "\n" + class_source
        return full_source

# --- Gradio UI and Logic ---

def run_and_submit_all(api_url: str, username: str):
    """
    Fetches all questions, runs the BasicAgent on them, submits all answers,
    and displays the results.
    """
    if not api_url:
        return "Please enter the API URL.", None # Status, DataFrame
    if not username:
        return "Please enter your Hugging Face username.", None # Status, DataFrame

    api_url = api_url.strip('/')
    questions_url = f"{api_url}/questions"
    submit_url = f"{api_url}/submit"

    # 1. Instantiate the Agent
    try:
        agent = BasicAgent()
        agent_code = agent.__repr__()
        # print(f"Agent Code (first 200): {agent_code[:200]}...") # Debug
    except Exception as e:
        print(f"Error instantiating agent or getting repr: {e}")
        return f"Error initializing agent: {e}", None

    # 2. Fetch All Questions
    print(f"Fetching questions from: {questions_url}")
    try:
        response = requests.get(questions_url, timeout=15)
        response.raise_for_status()
        questions_data = response.json()
        if not questions_data:
             return "Fetched questions list is empty.", None
        print(f"Fetched {len(questions_data)} questions.")
        status_update = f"Fetched {len(questions_data)} questions. Running agent..."
        # Yield intermediate status if using gr.update
    except requests.exceptions.RequestException as e:
        print(f"Error fetching questions: {e}")
        return f"Error fetching questions: {e}", None
    except Exception as e:
        print(f"An unexpected error occurred fetching questions: {e}")
        return f"An unexpected error occurred fetching questions: {e}", None

    # 3. Run Agent on Each Question
    results_log = [] # To store data for the results table
    answers_payload = [] # To store data for the submission API
    for item in questions_data:
        task_id = item.get("task_id")
        question_text = item.get("question")

        if not task_id or question_text is None:
            print(f"Skipping item with missing task_id or question: {item}")
            continue

        try:
            submitted_answer = agent(question_text) # Call the agent's logic
            answers_payload.append({
                "task_id": task_id,
                "submitted_answer": submitted_answer
            })
            results_log.append({
                "Task ID": task_id,
                "Question": question_text,
                "Submitted Answer": submitted_answer
            })
        except Exception as e:
             print(f"Error running agent on task {task_id}: {e}")
             # Decide how to handle agent errors - skip? submit default?
             # Here, we'll just log and potentially skip submission for this task if needed
             results_log.append({
                "Task ID": task_id,
                "Question": question_text,
                "Submitted Answer": f"AGENT ERROR: {e}"
            })


    if not answers_payload:
        return "Agent did not produce any answers to submit.", pd.DataFrame(results_log)

    # 4. Prepare Submission
    submission_data = {
        "username": username.strip(),
        "agent_code": agent_code,
        "answers": answers_payload
    }
    status_update = f"Agent finished. Submitting {len(answers_payload)} answers..."
    print(status_update)

    # 5. Submit to Leaderboard
    print(f"Submitting {len(answers_payload)} answers to: {submit_url}")
    try:
        response = requests.post(submit_url, json=submission_data, timeout=45) # Increased timeout
        response.raise_for_status()
        result_data = response.json()

        # Prepare final status message and results table
        final_status = (
            f"Submission Successful!\n"
            f"User: {result_data.get('username')}\n"
            f"Overall Score: {result_data.get('score')}% "
            f"({result_data.get('correct_count')}/{result_data.get('total_attempted')} correct)\n"
            f"Message: {result_data.get('message')}"
        )
        print("Submission successful.")
        results_df = pd.DataFrame(results_log)
        return final_status, results_df

    except requests.exceptions.HTTPError as e:
        error_detail = e.response.text
        try:
            error_json = e.response.json()
            error_detail = error_json.get('detail', error_detail)
        except requests.exceptions.JSONDecodeError:
            pass
        status_message = f"Submission Failed (HTTP {e.response.status_code}): {error_detail}"
        print(status_message)
        results_df = pd.DataFrame(results_log) # Show attempts even if submission failed
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
        "Enter the API URL and your username, then click Run. "
        "This will fetch all questions, run the *very basic* agent on them, "
        "submit all answers at once, and display the results."
    )

    with gr.Row():
        api_url_input = gr.Textbox(label="FastAPI API URL", value=DEFAULT_API_URL)
        hf_username_input = gr.Textbox(label="Hugging Face Username")

    run_button = gr.Button("Run Evaluation & Submit All Answers")

    status_output = gr.Textbox(label="Run Status / Submission Result", lines=4, interactive=False)
    results_table = gr.DataFrame(label="Questions and Agent Answers", wrap=True)

    # --- Component Interaction ---
    run_button.click(
        fn=run_and_submit_all,
        inputs=[api_url_input, hf_username_input],
        outputs=[status_output, results_table]
    )

if __name__ == "__main__":
    print("Launching Gradio Interface for Basic Agent Evaluation...")
    demo.launch(debug=True)