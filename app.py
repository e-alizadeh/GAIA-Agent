import os
import gradio as gr
from gradio import ChatMessage
import requests # Although not used in dummy tools, keep if future tools need it
from typing import Dict, List, Generator, Sequence
from langchain_core.messages import HumanMessage, BaseMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent
import logging

# Configure logging for better debugging on Spaces
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Tool Definitions ---
@tool
def get_lat_lng(location_description: str) -> dict[str, float]:
    """Get the latitude and longitude of a location description (e.g., 'Paris', 'Tokyo, Japan')."""
    # In a real app, you'd call a geocoding API here.
    logger.info(f"Tool 'get_lat_lng' called with location: {location_description}")
    # Dummy response for demonstration
    if "london" in location_description.lower():
        return {"lat": 51.5074, "lng": -0.1278}
    elif "tokyo" in location_description.lower():
         return {"lat": 35.6895, "lng": 139.6917}
    elif "paris" in location_description.lower():
         return {"lat": 48.8566, "lng": 2.3522}
    elif "new york" in location_description.lower():
         return {"lat": 40.7128, "lng": -74.0060}
    else:
        # Default dummy response
        return {"lat": 51.1, "lng": -0.1}

@tool
def get_weather(lat: float, lng: float) -> dict[str, str]:
    """Get the current weather conditions at a specific latitude and longitude."""
    # In a real app, you'd call a weather API (e.g., OpenWeatherMap) here.
    logger.info(f"Tool 'get_weather' called with lat: {lat}, lng: {lng}")
    # Dummy response based on latitude for variety
    if 40 < lat < 50: # Approx Paris/New York
        return {"temperature": "18°C", "description": "Cloudy"}
    elif lat > 50: # Approx London
        return {"temperature": "15°C", "description": "Rainy"}
    else: # Approx Tokyo / Default
        return {"temperature": "25°C", "description": "Sunny"}

# --- Agent and Streaming Logic ---
def initialize_agent():
    """Initializes the LangChain agent."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        logger.error("OPENAI_API_KEY environment variable not set.")
        # Option 1: Raise an error to stop the app
        # raise ValueError("OpenAI API Key not found. Please set it in the Space secrets.")
        # Option 2: Return None and handle it in the stream function
        return None

    try:
        llm = ChatOpenAI(temperature=0, model="gpt-4", openai_api_key=api_key)
        # Note: MemorySaver() is in-memory. State will be lost on space restarts/sleeps.
        # For persistent memory across sessions/restarts, you'd need a persistent checkpointer (e.g., using Redis, SQL).
        memory = MemorySaver()
        tools = [get_lat_lng, get_weather]
        agent_executor = create_react_agent(llm, tools, checkpointer=memory)
        logger.info("LangChain agent initialized successfully.")
        return agent_executor
    except Exception as e:
        logger.error(f"Failed to initialize LangChain agent: {e}", exc_info=True)
        return None

# Initialize agent once when the script starts
agent_executor = initialize_agent()

# Define the streaming function for Gradio ChatInterface
def stream_from_agent(message: str, history: List[List[str]]) -> Generator[Sequence[ChatMessage], None, None]:
    """
    Processes user messages through the LangChain agent, yielding intermediate steps.

    Args:
        message: The user's input message.
        history: The conversation history provided by Gradio (list of [user, assistant] pairs).

    Yields:
        A sequence of Gradio ChatMessage objects representing the agent's thoughts and actions.
    """
    global agent_executor # Use the globally initialized agent

    if agent_executor is None:
        error_msg = "Agent initialization failed. Please check the logs and ensure the OPENAI_API_KEY secret is set correctly."
        yield [ChatMessage(role="assistant", content=error_msg)]
        return

    logger.info(f"Received message: {message}")
    logger.info(f"History: {history}")

    # Convert Gradio history to LangChain message format
    # Note: create_react_agent expects a list of BaseMessages under the "messages" key.
    # It typically works best with a single HumanMessage as input per turn for the ReAct loop.
    # We will use the memory checkpointer to handle history persistence within the agent's context.
    langchain_message = HumanMessage(content=message)

    messages_to_display: List[ChatMessage] = []
    final_response_content = ""

    try:
        # Note: Using a fixed thread_id means all users share the same memory state if MemorySaver is used.
        # For isolated user sessions, you'd need a mechanism to generate/retrieve unique thread_ids per user/session.
        # This often requires integrating with Gradio's state or session management.
        # For simplicity here, we use a fixed ID as in the original code.
        thread_id = "shared_weather_thread_123"
        config = {"configurable": {"thread_id": thread_id}}

        # Stream the agent's execution steps
        for chunk in agent_executor.stream({"messages": [langchain_message]}, config=config):
            logger.debug(f"Agent chunk received: {chunk}") # Use debug level for verbose chunk logging

            # Check for Agent Actions (Tool Calls)
            if agent_action := chunk.get("agent"):
                 # Often the agent's rationale or decision to use a tool might be here
                 # Depending on the specific agent type, you might parse agent_action differently
                 if agent_action.get("messages"):
                      for msg in agent_action["messages"]:
                           if hasattr(msg, 'tool_calls') and msg.tool_calls:
                                for tool_call in msg.tool_calls:
                                    # Display the tool call intention
                                    tool_msg = ChatMessage(
                                        role="assistant", # Show tool usage as assistant action
                                        content=f"Parameters: `{tool_call['args']}`",
                                        metadata={
                                            "title": f"🛠️ Calling Tool: `{tool_call['name']}`",
                                            "tool_call_id": tool_call["id"], # Store ID to match response
                                        }
                                    )
                                    messages_to_display.append(tool_msg)
                                    yield messages_to_display
                           # Capture potential intermediate reasoning if available (depends on agent/LLM)
                           elif hasattr(msg, 'content') and isinstance(msg.content, str) and msg.content:
                               # Avoid displaying the *final* answer prematurely if it appears mid-stream
                               # The final answer is usually in the last chunk's 'agent' message list
                               pass # We'll handle the final answer specifically later


            # Check for Tool Execution Results
            if tool_chunk := chunk.get("tools"):
                 if tool_chunk.get("messages"):
                    for tool_response in tool_chunk["messages"]:
                        # Find the corresponding tool call message to update it
                        found = False
                        for i, msg in enumerate(messages_to_display):
                            if msg.metadata and msg.metadata.get("tool_call_id") == tool_response.tool_call_id:
                                # Update the existing tool message with the result
                                updated_content = msg.content + f"\nResult: `{tool_response.content}`"
                                messages_to_display[i] = ChatMessage(
                                    role=msg.role,
                                    content=updated_content,
                                    metadata=msg.metadata # Keep original metadata
                                )
                                found = True
                                break
                        if found:
                            yield messages_to_display
                        else:
                             # If matching call not found (shouldn't happen often), display separately
                             tool_result_msg = ChatMessage(
                                 role="tool", # Or keep as assistant? 'tool' role might not render well by default
                                 content=f"Tool Result (`{tool_response.tool_call_id}`): `{tool_response.content}`"
                             )
                             messages_to_display.append(tool_result_msg)
                             yield messages_to_display


            # Check for the Final Agent Response
            # The final answer is typically the last message in the 'agent' chunk's list
            if agent_final := chunk.get("agent"):
                if agent_final.get("messages"):
                    last_message = agent_final["messages"][-1]
                    # Ensure it's the final response (often not a tool call)
                    if hasattr(last_message, 'content') and not (hasattr(last_message, 'tool_calls') and last_message.tool_calls):
                        final_response_content = last_message.content


        # After the loop, ensure the final response is added if it hasn't been implicitly handled
        if final_response_content:
            # Check if the last displayed message is already the final response
            is_already_displayed = False
            if messages_to_display:
                last_displayed = messages_to_display[-1]
                # Simple check: if last displayed message has no tool metadata and content matches
                if not (last_displayed.metadata and "tool_call_id" in last_displayed.metadata) and last_displayed.content == final_response_content:
                     is_already_displayed = True

            if not is_already_displayed:
                final_msg = ChatMessage(role="assistant", content=final_response_content)
                messages_to_display.append(final_msg)
                yield messages_to_display
        elif not messages_to_display:
            # Handle cases where the agent might not produce a final response (e.g., errors)
             yield [ChatMessage(role="assistant", content="Sorry, I couldn't process that request.")]


    except Exception as e:
        logger.error(f"Error during agent stream: {e}", exc_info=True)
        error_message = f"An error occurred: {e}"
        yield [ChatMessage(role="assistant", content=error_message)]


# --- Gradio Interface Definition ---
# Use gr.ChatInterface with type="messages" for full ChatMessage object support
demo = gr.ChatInterface(
    fn=stream_from_agent,
    chatbot=gr.Chatbot(
        bubble_full_width=False,
        show_copy_button=True,
        render=False # Render manually for better control if needed, but False is fine here
        ),
    input_components=[gr.Textbox(label="Ask the weather assistant")], # Customize input textbox
    # `type="messages"` passes message/history using gr.ChatMessage objects (needed for metadata)
    # However, ChatInterface's standard history format is List[List[str]].
    # Let's stick to the standard fn signature for ChatInterface if possible
    # and convert history inside the function if needed.
    # Reverting fn signature slightly based on typical ChatInterface usage.
    # If type="messages" is used, fn signature might expect different types.
    # Sticking to standard List[List[str]] history for compatibility.
    # Let's adjust the stream_from_agent function signature slightly if needed.
    # **Correction**: `gr.ChatInterface` *does* handle the `List[List[str]]` history format even when yielding `ChatMessage`.
    # The function signature `(message: str, history: List[List[str]])` is correct.

    title="🌤️ Weather Assistant with LangGraph ReAct Agent",
    description="Ask about the weather anywhere! Watch the agent think step-by-step as it uses tools.",
    examples=[
        ["What's the weather like in Tokyo?"],
        ["Is it sunny in Paris right now?"],
        ["Should I bring an umbrella in New York today?"]
    ],
    cache_examples=False, # Disable caching for dynamic examples if needed
    theme="soft", # Optional: Apply a theme
    retry_btn=None, # Disable retry button if stream handles errors
    undo_btn="Delete Previous", # Customize undo button text
    clear_btn="Clear Conversation", # Customize clear button text
)

# --- Launch the App ---
if __name__ == "__main__":
    # Launch the Gradio app
    # share=False is default and recommended for Spaces
    # debug=True can be helpful during development but disable for production
    # server_name="0.0.0.0" allows access within the Space's network
    demo.launch(server_name="0.0.0.0", server_port=7860)