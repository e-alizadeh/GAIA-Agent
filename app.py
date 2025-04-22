import os
import gradio as gr
import requests
import inspect # To get source code for __repr__
import asyncio
from typing import Dict, List, AsyncGenerator, Union, Tuple, Optional

# --- LangChain Specific Imports ---
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent

# --- Constants ---
DEFAULT_API_URL = "http://127.0.0.1:8000" # Default URL for your FastAPI app

# --- Tools (Keep these defined globally or ensure they are included in __repr__) ---
@tool
def get_lat_lng(location_description: str) -> dict[str, float]:
    """Get the latitude and longitude of a location."""
    print(f"Tool: Getting lat/lng for {location_description}")
    # Replace with actual API call in a real app
    if "tokyo" in location_description.lower():
        return {"lat": 35.6895, "lng": 139.6917}
    elif "paris" in location_description.lower():
         return {"lat": 48.8566, "lng": 2.3522}
    elif "new york" in location_description.lower():
        return {"lat": 40.7128, "lng": -74.0060}
    else:
        return {"lat": 51.5072, "lng": -0.1276} # Default London

@tool
def get_weather(lat: float, lng: float) -> dict[str, str]:
    """Get the weather at a location."""
    print(f"Tool: Getting weather for lat={lat}, lng={lng}")
    # Replace with actual API call in a real app
    if lat > 45: # Northern locations
        return {"temperature": "15°C", "description": "Cloudy"}
    elif lat > 30: # Mid locations
        return {"temperature": "25°C", "description": "Sunny"}
    else: # Southern locations
        return {"temperature": "30°C", "description": "Very Sunny"}

# --- Agent Class Definition ---
class MyLangChainAgent:
    """
    A sample LangChain agent class designed for interaction and submission.
    NOTE: The current tools (weather/location) are placeholders and WILL NOT
          correctly answer GAIA benchmark questions. This class structure
          demonstrates how to integrate an agent with the submission API.
          Replace LLM, tools, and potentially the agent type for actual GAIA tasks.
    """
    def __init__(self, model_name="gpt-4", temperature=0):
        # Ensure API key is available
        if not os.getenv("OPENAI_API_KEY"):
            raise ValueError("OPENAI_API_KEY environment variable not set.")

        self.llm = ChatOpenAI(temperature=temperature, model=model_name)
        self.tools = [get_lat_lng, get_weather] # Use the globally defined tools
        self.memory = MemorySaver()
        # Create the agent executor
        self.agent_executor = create_react_agent(self.llm, self.tools, checkpointer=self.memory)
        print("MyLangChainAgent initialized.")

    async def __call__(self, question: str, thread_id: str) -> AsyncGenerator[Union[str, Dict[str, str]], str]:
        """
        Runs the agent asynchronously, yielding intermediate steps and returning the final answer.

        Args:
            question: The input question string.
            thread_id: A unique identifier for the conversation thread.

        Yields:
            Intermediate steps (tool calls/results) as strings or dicts.

        Returns:
            The final AI answer as a string.
        """
        print(f"Agent executing for thread_id: {thread_id} on question: {question[:50]}...")
        lc_messages: List[BaseMessage] = [HumanMessage(content=question)]
        final_answer = ""
        full_response_content = "" # Store the complete AI response chunks

        async for chunk in self.agent_executor.astream_events(
            {"messages": lc_messages},
            config={"configurable": {"thread_id": thread_id}},
            version="v1"
        ):
            event = chunk["event"]
            data = chunk["data"]
            # print(f"DEBUG: Event: {event}, Data Keys: {data.keys()}") # Debugging line

            if event == "on_chat_model_stream":
                content = data["chunk"].content
                if content:
                    # print(f"DEBUG: AI Chunk: {content}") # Debugging line
                    full_response_content += content
                    # Yield potentially incomplete response for live typing effect if needed
                    # yield {"type": "stream", "content": content }

            elif event == "on_tool_start":
                tool_input_str = str(data.get('input', ''))
                yield f"🛠️ Using tool: **{data['name']}** with input: `{tool_input_str}`"

            elif event == "on_tool_end":
                 tool_output_str = str(data.get('output', ''))
                 yield f"✅ Tool **{data['name']}** finished.\nResult: `{tool_output_str}`"
            
            # Detect the end of the conversation turn (heuristic)
            # The 'on_chain_end' event for the top-level graph might signal the end.
            # Or check the 'messages' list in the final state if available.
            # For create_react_agent, the final AIMessage is often the last main event.
            # We will capture the last full AI message content after the loop.

        # After iterating through all chunks, the final answer should be in full_response_content
        final_answer = full_response_content.strip()
        print(f"Agent execution finished. Final Answer: {final_answer[:100]}...")
        # Yield the complete final answer distinctly if needed
        # yield {"type": "final_answer_marker", "content": final_answer} # Example marker
        return final_answer # Return the final answer

    def __repr__(self) -> str:
        """
        Return the source code required to reconstruct this agent, including
        the class definition, tool functions, and necessary imports.
        """
        imports = [
            "import os",
            "from typing import Dict, List, AsyncGenerator, Union, Tuple, Optional",
            "from langchain_core.messages import HumanMessage, AIMessage, BaseMessage",
            "from langchain_core.tools import tool",
            "from langchain_openai import ChatOpenAI",
            "from langgraph.checkpoint.memory import MemorySaver",
            "from langgraph.prebuilt import create_react_agent",
            "import inspect", # Needed if repr itself uses inspect dynamically
            "import asyncio", # Needed for async call
            "\n"
        ]
        # Get source code of tool functions
        tool_sources = []
        for t in self.tools:
             try:
                 tool_sources.append(inspect.getsource(t))
             except (TypeError, OSError) as e:
                  print(f"Warning: Could not get source for tool {t.__name__}: {e}")
                  tool_sources.append(f"# Could not automatically get source for tool: {t.__name__}\n")

        # Get source code of the class itself
        class_source = inspect.getsource(MyLangChainAgent)

        # Combine imports, tools, and class definition
        full_source = "\n".join(imports) + "\n\n" + \
                      "\n\n".join(tool_sources) + "\n\n" + \
                      class_source
        return full_source


# --- Gradio UI and Logic ---

# Initialize the agent (do this once outside the request functions)
# Handle potential API key error during initialization
try:
    agent_instance = MyLangChainAgent()
except ValueError as e:
    print(f"ERROR initializing agent: {e}")
    # Provide a dummy agent or exit if critical
    agent_instance = None # Or raise SystemExit("Agent initialization failed")

def format_chat_history(history: List[List[Optional[str]]]) -> List[Tuple[Optional[str], Optional[str]]]:
    """Helper to format Gradio history for display."""
    # Gradio's history format is List[List[user_msg | None, ai_msg | None]]
    # We want List[Tuple[user_msg | None, ai_msg | None]] for Chatbot
    formatted = []
    for turn in history:
        formatted.append(tuple(turn))
    return formatted


async def fetch_and_display_question(api_url: str):
    """Calls the backend to get a random question."""
    if not api_url:
        return "Please enter the API URL.", "", "", gr.update(value=""), gr.update(value="") # Clear chat too
    
    question_url = f"{api_url.strip('/')}/random-question"
    print(f"Fetching question from: {question_url}")
    try:
        response = requests.get(question_url, timeout=10)
        response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)
        data = response.json()
        task_id = data.get("task_id")
        question_text = data.get("question")
        if task_id and question_text:
            print(f"Fetched Task ID: {task_id}")
            # Return updates for Gradio components: Status, Task ID, Question Text, Clear Agent Answer, Clear Chat
            return "Question fetched successfully!", task_id, question_text, "", [] # Clears answer and chat history
        else:
            return "Error: Invalid data format received from API.", "", "", "", []
    except requests.exceptions.RequestException as e:
        print(f"Error fetching question: {e}")
        return f"Error fetching question: {e}", "", "", "", []
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return f"An unexpected error occurred: {e}", "", "", "", []


async def run_agent_interaction(
    message: str,
    history: List[List[Optional[str]]],
    current_task_id: str,
    # agent_instance: MyLangChainAgent # Agent passed via state potentially
):
    """Handles the chat interaction, runs the agent, yields steps, updates final answer state."""
    if agent_instance is None:
         yield "Agent not initialized. Please check API keys and restart."
         return

    if not current_task_id:
        yield "Please fetch a question first using the button above."
        return

    # The 'message' here is the user's latest input in the chat.
    # For this workflow, we assume the main input is the fetched question.
    # We'll use the fetched question (implicitly stored) to run the agent.
    # If you want interactive chat *about* the question, the logic needs adjustment.

    # For simplicity, let's assume the user's message *is* the question or a prompt related to it.
    # In the GAIA context, usually, the agent just runs on the provided question directly.
    # We'll use the `current_task_id` to generate a unique thread_id for LangGraph memory.
    thread_id = f"gaia_task_{current_task_id}_{os.urandom(4).hex()}"

    print(f"Running agent for user message: {message[:50]}...")
    history.append([message, None]) # Add user message to history

    final_agent_answer = None
    full_yielded_response = ""

    # Use the agent's __call__ method
    async for step in agent_instance(message, thread_id=thread_id):
        if isinstance(step, str):
            # Intermediate step (tool call, result, maybe stream chunk)
            history[-1][1] = step # Update the AI's response in the last turn
            yield format_chat_history(history) # Update chatbot UI
            full_yielded_response = step # Track last yielded message
        # If __call__ yielded dicts for streaming, handle here:
        # elif isinstance(step, dict) and step.get("type") == "stream":
        #     history[-1][1] = (history[-1][1] or "") + step["content"]
        #     yield format_chat_history(history)

    # After the loop, the `step` variable holds the return value (final answer)
    final_agent_answer = step
    print(f"Agent final answer received: {final_agent_answer[:100]}...")

    # Update the history with the definitive final answer
    if final_agent_answer:
        history[-1][1] = final_agent_answer # Replace intermediate steps with final one
    elif full_yielded_response:
        # Fallback if final answer wasn't returned correctly but we yielded something
         history[-1][1] = full_yielded_response
         final_agent_answer = full_yielded_response # Use the last yielded message as answer
    else:
         history[-1][1] = "Agent did not produce a final answer."
         final_agent_answer = "" # Ensure it's a string

    # Yield the final state of the history and update the hidden state for the final answer
    yield format_chat_history(history), final_agent_answer


def submit_to_leaderboard(
    api_url: str,
    username: str,
    task_id: str,
    agent_answer: str,
    # agent_instance: MyLangChainAgent # Pass agent via state if needed
):
    """Submits the agent's answer and code to the FastAPI backend."""
    if agent_instance is None:
        return "Agent not initialized. Cannot submit."
    if not api_url:
        return "Please enter the API URL."
    if not username:
        return "Please enter your Hugging Face username."
    if not task_id:
        return "No task ID available. Please fetch a question first."
    if agent_answer is None or agent_answer.strip() == "": # Check if None or empty
        # Maybe allow submission of empty answer? Depends on requirements.
        print("Warning: Submitting empty answer.")
        # return "Agent has not provided an answer yet."


    submit_url = f"{api_url.strip('/')}/submit"
    print(f"Submitting to: {submit_url}")

    # Get agent code
    try:
        agent_code = agent_instance.__repr__()
        # print(f"Agent Code (first 200 chars):\n{agent_code[:200]}...") # Debug
    except Exception as e:
        print(f"Error getting agent representation: {e}")
        return f"Error generating agent code for submission: {e}"

    # Prepare submission data according to Pydantic model in FastAPI
    submission_data = {
        "username": username.strip(),
        "agent_code": agent_code,
        "answers": [
            {
                "task_id": task_id,
                "submitted_answer": agent_answer # Use the stored final answer
            }
            # Add more answers here if submitting a batch
        ]
    }

    try:
        response = requests.post(submit_url, json=submission_data, timeout=30)
        response.raise_for_status()
        result_data = response.json()
        # Format the result nicely for display
        result_message = (
            f"Submission Successful!\n"
            f"User: {result_data.get('username')}\n"
            f"Score: {result_data.get('score')}\n"
            f"Correct: {result_data.get('correct_count')}/{result_data.get('total_attempted')}\n"
            f"Message: {result_data.get('message')}\n"
            f"Timestamp: {result_data.get('timestamp')}"
        )
        print("Submission successful.")
        return result_message
    except requests.exceptions.HTTPError as e:
        # Try to get detail from response body if available
        error_detail = e.response.text
        try:
            error_json = e.response.json()
            error_detail = error_json.get('detail', error_detail)
        except requests.exceptions.JSONDecodeError:
            pass # Keep the raw text if not JSON
        print(f"HTTP Error during submission: {e.response.status_code} - {error_detail}")
        return f"Submission Failed (HTTP {e.response.status_code}): {error_detail}"
    except requests.exceptions.RequestException as e:
        print(f"Network error during submission: {e}")
        return f"Submission Failed: Network error - {e}"
    except Exception as e:
        print(f"An unexpected error occurred during submission: {e}")
        return f"Submission Failed: An unexpected error occurred - {e}"


# --- Build Gradio Interface using Blocks ---
with gr.Blocks() as demo:
    gr.Markdown("# Agent Evaluation Interface")
    gr.Markdown(
        "Fetch a random question from the evaluation API, interact with the agent "
        "(Note: the default agent answers weather questions, not GAIA), "
        "and submit the agent's final answer to the leaderboard."
    )

    # --- State Variables ---
    # Store current task info, agent's final answer, and the agent instance
    current_task_id = gr.State("")
    current_question_text = gr.State("")
    current_agent_answer = gr.State("") # Stores the final answer string from the agent
    # agent_state = gr.State(agent_instance) # Pass agent instance via state

    with gr.Row():
        api_url_input = gr.Textbox(label="FastAPI API URL", value=DEFAULT_API_URL)
        hf_username_input = gr.Textbox(label="Hugging Face Username")

    with gr.Row():
        fetch_button = gr.Button("Get Random Question")
        submission_status_display = gr.Textbox(label="Status", interactive=False) # For fetch status

    with gr.Row():
        question_display = gr.Textbox(label="Current Question", lines=3, interactive=False)

    gr.Markdown("---")
    gr.Markdown("## Agent Interaction")

    chatbot = gr.Chatbot(label="Agent Conversation", height=400)
    msg_input = gr.Textbox(label="Send a message to the Agent (or just observe)") # Input for chat

    # Hidden Textbox to display the final extracted answer (optional, for clarity)
    final_answer_display = gr.Textbox(label="Agent's Final Answer (Extracted)", interactive=False)

    gr.Markdown("---")
    gr.Markdown("## Submission")
    with gr.Row():
         submit_button = gr.Button("Submit Current Answer to Leaderboard")

    submission_result_display = gr.Markdown(label="Submission Result", value="*Submit an answer to see the result here.*") # Use Markdown for better formatting


    # --- Component Interactions ---

    # Fetch Button Action
    fetch_button.click(
        fn=fetch_and_display_question,
        inputs=[api_url_input],
        outputs=[
            submission_status_display, # Shows fetch status
            current_task_id,           # Updates hidden state
            question_display,          # Updates question text box
            final_answer_display,      # Clears old final answer
            chatbot                    # Clears chat history
        ]
    )

    # Chat Submission Action (when user sends message in chat)
    msg_input.submit(
        fn=run_agent_interaction,
        inputs=[
            msg_input,                  # User message from chat input
            chatbot,                    # Current chat history
            current_task_id,            # Current task ID from state
            # agent_state               # Pass agent instance state
        ],
        outputs=[
            chatbot,                    # Updated chat history
            current_agent_answer        # Update the hidden state holding the final answer
        ]
    ).then(
        # After agent runs, update the visible "Final Answer" box from the state
        lambda answer_state: answer_state,
        inputs=[current_agent_answer],
        outputs=[final_answer_display]
    )

    # Clear message input after submission
    msg_input.submit(lambda: "", None, msg_input, queue=False)


    # Submit Button Action
    submit_button.click(
        fn=submit_to_leaderboard,
        inputs=[
            api_url_input,
            hf_username_input,
            current_task_id,
            current_agent_answer,       # Use the stored final answer state
            # agent_state               # Pass agent instance state
        ],
        outputs=[submission_result_display] # Display result message
    )


if __name__ == "__main__":
    if agent_instance is None:
        print("\nFATAL: Agent could not be initialized. Gradio app will not run correctly.")
        print("Please ensure OPENAI_API_KEY is set and valid.\n")
        # Optionally exit here if agent is critical
        # exit(1)
    else:
        print("Launching Gradio Interface...")
        demo.launch(debug=True, server_name="0.0.0.0") # Share=False by default for security