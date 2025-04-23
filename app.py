import os
import gradio as gr
import requests
import inspect # To get source code for __repr__
import pandas as pd

# --- Constants ---
DEFAULT_API_URL = "https://jofthomas-unit4-scoring.hf.space/" # Default URL for your FastAPI app

# --- Basic Agent Definition ---
## This is where you should implement your own agent and tools

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

    # __repr__ seems intended to get the *source* code, not just representation
    # Let's keep it but note that get_current_script_content might be more robust
    # if the class definition changes significantly or relies on external state.
    def __repr__(self) -> str:
        """
        Return the source code required to reconstruct this agent.
        NOTE: This might be brittle. Using get_current_script_content is likely safer.
        """
        imports = [
            "import inspect\n"
        ]
        try:
            class_source = inspect.getsource(BasicAgent)
            full_source = "\n".join(imports) + "\n" + class_source
            return full_source
        except Exception as e:
            print(f"Error getting source code via inspect: {e}")
            return f"# Could not get source via inspect: {e}"

# --- Gradio UI and Logic ---
def get_current_script_content() -> str:
    """Attempts to read and return the content of the currently running script."""
    try:
        # __file__ holds the path to the current script
        script_path = os.path.abspath(__file__)
        print(f"Reading script content from: {script_path}")
        with open(script_path, 'r', encoding='utf-8') as f:
            return f.read()
    except NameError:
        # __file__ is not defined (e.g., running in an interactive interpreter or frozen app)
        print("Warning: __file__ is not defined. Cannot read script content this way.")
        # Fallback or alternative method could be added here if needed
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
    # --- Determine HF Space URL and Print Environment Info ---
    space_host = os.getenv("SPACE_HOST")
    hf_space_url = "Runtime: Locally or unknown environment (SPACE_HOST env var not found)"
    if space_host:
         # Construct the standard URL format for HF Spaces
         hf_space_url = f"Runtime: Hugging Face Space (https://{space_host}.hf.space)"

    # Print runtime info at the start of the function execution
    print("\n" + "="*60)
    print("Executing run_and_submit_all function...")
    print(hf_space_url) # Print the determined runtime URL
    # --- End Environment Info ---

    if profile:
        username= f"{profile.username}"
        print(f"User logged in: {username}")
    else:
        print("User not logged in.")
        print("="*60 + "\n") # Close the separator block
        return "Please Login to Hugging Face with the button.", None # Return early

    print("="*60 + "\n") # Separator after initial checks if logged in

    api_url = DEFAULT_API_URL
    questions_url = f"{api_url}/questions"
    submit_url = f"{api_url}/submit"

    # 1. Instantiate the Agent
    try:
        agent = BasicAgent()
        # Using get_current_script_content() is likely more reliable for submission
        # agent_code = agent.__repr__() # Keep if needed, but prefer file content
        # print(f"Agent Code via __repr__ (first 200): {agent_code[:200]}...") # Debug
    except Exception as e:
        print(f"Error instantiating agent: {e}")
        return f"Error initializing agent: {e}", None

    # Get agent code by reading the current script file - generally more robust
    agent_code = get_current_script_content()
    if agent_code.startswith("# Agent code unavailable"):
        print("Warning: Using potentially incomplete agent code due to reading error.")
        # Optional: Fall back to agent.__repr__() if needed
        # agent_code = agent.__repr__()

    # 2. Fetch All Questions
    print(f"Fetching questions from: {questions_url}")
    try:
        response = requests.get(questions_url, timeout=15)
        response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)
        questions_data = response.json()
        if not questions_data:
             print("Fetched questions list is empty.")
             return "Fetched questions list is empty or invalid format.", None
        print(f"Fetched {len(questions_data)} questions.")
        # status_update = f"Fetched {len(questions_data)} questions. Running agent..." # For yield/streaming
    except requests.exceptions.RequestException as e:
        print(f"Error fetching questions: {e}")
        return f"Error fetching questions: {e}", None
    except requests.exceptions.JSONDecodeError as e:
         print(f"Error decoding JSON response from questions endpoint: {e}")
         print(f"Response text: {response.text[:500]}") # Log response text for debugging
         return f"Error decoding server response for questions: {e}", None
    except Exception as e: # Catch other potential errors
        print(f"An unexpected error occurred fetching questions: {e}")
        return f"An unexpected error occurred fetching questions: {e}", None

    # 3. Run Agent on Each Question
    results_log = [] # To store data for the results table
    answers_payload = [] # To store data for the submission API
    print(f"Running agent on {len(questions_data)} questions...")
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
             results_log.append({
                "Task ID": task_id,
                "Question": question_text,
                "Submitted Answer": f"AGENT ERROR: {e}"
            })
             # Decide if you want to submit agent errors or skip:
             # answers_payload.append({"task_id": task_id, "submitted_answer": f"AGENT ERROR: {e}"})

    if not answers_payload:
        print("Agent did not produce any answers to submit.")
        # Still show results log even if nothing submitted
        return "Agent did not produce any answers to submit.", pd.DataFrame(results_log)

    # 4. Prepare Submission
    submission_data = {
        "username": username.strip(),
        "agent_code": agent_code, # Using the code read from file
        "answers": answers_payload
    }
    status_update = f"Agent finished. Submitting {len(answers_payload)} answers for user '{username}'..."
    print(status_update)

    # 5. Submit to Leaderboard
    print(f"Submitting {len(answers_payload)} answers to: {submit_url}")
    try:
        # Ensure submission_data is serializable, agent_code should be string
        response = requests.post(submit_url, json=submission_data, timeout=60) # Increased timeout further
        response.raise_for_status()
        result_data = response.json()

        # Prepare final status message and results table
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
            # Try to get more specific error detail from JSON response body
            error_json = e.response.json()
            error_detail += f" Detail: {error_json.get('detail', e.response.text)}"
        except requests.exceptions.JSONDecodeError:
            # If response is not JSON, use the raw text
            error_detail += f" Response: {e.response.text[:500]}" # Limit length
        status_message = f"Submission Failed: {error_detail}"
        print(status_message)
        results_df = pd.DataFrame(results_log) # Show attempts even if submission failed
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
    except Exception as e: # Catch unexpected errors during submission phase
        status_message = f"An unexpected error occurred during submission: {e}"
        print(status_message)
        results_df = pd.DataFrame(results_log)
        return status_message, results_df


# --- Build Gradio Interface using Blocks ---
with gr.Blocks() as demo:
    gr.Markdown("# Basic Agent Evaluation Runner")
    gr.Markdown(
        "Please clone this space, then modify the code to define your agent's logic within the `BasicAgent` class. " # Clarified instructions
        "Log in to your Hugging Face account using the button below. This uses your HF username for submission. "
        "Click 'Run Evaluation & Submit All Answers' to fetch questions, run your agent, submit answers, and see the score."
    )

    gr.LoginButton()

    run_button = gr.Button("Run Evaluation & Submit All Answers")

    status_output = gr.Textbox(label="Run Status / Submission Result", lines=5, interactive=False) # Increased lines
    results_table = gr.DataFrame(label="Questions and Agent Answers", wrap=True, max_rows=10) # Added max_rows

    # --- Component Interaction ---
    # Use the profile information directly from the LoginButton state (implicitly passed)
    run_button.click(
        fn=run_and_submit_all,
        # Input is implicitly the profile data from LoginButton state
        outputs=[status_output, results_table]
    )

if __name__ == "__main__":
    print("\n" + "-"*30 + " App Starting " + "-"*30)
    # Check for SPACE_HOST at startup for information
    space_host_startup = os.getenv("SPACE_HOST")
    if space_host_startup:
        print(f"✅ SPACE_HOST found: {space_host_startup}")
        print(f"   App should be available at: https://{space_host_startup}.hf.space")
    else:
        print("ℹ️  SPACE_HOST environment variable not found (running locally or not on standard HF Space runtime).")
        print("   App will likely be available at local URLs printed by Gradio below.")
    print("-"*(60 + len(" App Starting ")) + "\n")

    print("Launching Gradio Interface for Basic Agent Evaluation...")
    # Set share=False as the primary access point is the HF Space URL
    demo.launch(debug=True, share=False)