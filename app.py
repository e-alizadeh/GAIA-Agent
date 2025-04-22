import os
import gradio as gr
# Keep using gradio.ChatMessage for type hints if needed, but not for yielding complex structures directly to ChatInterface
# from gradio import ChatMessage # Maybe remove this import if not used elsewhere
import requests
from typing import Dict, List, AsyncGenerator, Union, Tuple
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage # Use LangChain messages internally
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent

# --- Tools remain the same ---
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
    # Dummy logic based on lat
    if lat > 45: # Northern locations
        return {"temperature": "15°C", "description": "Cloudy"}
    elif lat > 30: # Mid locations
        return {"temperature": "25°C", "description": "Sunny"}
    else: # Southern locations
        return {"temperature": "30°C", "description": "Very Sunny"}


async def Answer_from_agent(message: str, history: List[List[str]]) -> AsyncGenerator[str, None]:
    """Processes message through LangChain agent, yielding intermediate steps as strings."""

    # Convert Gradio history to LangChain messages
    lc_messages = []
    for user_msg, ai_msg in history:
        if user_msg:
            lc_messages.append(HumanMessage(content=user_msg))
        if ai_msg:
            # Important: Handle potential previous intermediate strings from AI
            # If the ai_msg contains markers like "🛠️ Using", it was an intermediate step.
            # For simplicity here, we assume full AI responses were stored previously.
            # A more robust solution might involve storing message types in history.
            if not ai_msg.startswith("🛠️ Using") and not ai_msg.startswith("Result:"):
                 lc_messages.append(AIMessage(content=ai_msg))

    lc_messages.append(HumanMessage(content=message))

    llm = ChatOpenAI(temperature=0, model="gpt-4")
    memory = MemorySaver() # Be mindful of memory state if agent is re-initialized every time
    tools = [get_lat_lng, get_weather]
    agent_executor = create_react_agent(llm, tools, checkpointer=memory)

    # Use a unique thread_id per session if needed, or manage state differently
    # Using a fixed one like "abc123" means all users share the same memory if server restarts aren't frequent
    thread_id = "user_session_" + str(os.urandom(4).hex()) # Example: generate unique ID

    full_response = ""

    async for chunk in agent_executor.astream_events(
        {"messages": lc_messages},
        config={"configurable": {"thread_id": thread_id}},
        version="v1" # Use v1 for events streaming
    ):
        event = chunk["event"]
        data = chunk["data"]

        if event == "on_chat_model_stream":
            content = data["chunk"].content
            if content:
                full_response += content
                yield full_response 

        elif event == "on_tool_start":
            tool_input_str = str(data.get('input', ''))
            yield f"🛠️ Using tool: **{data['name']}** with input: `{tool_input_str}`"

        elif event == "on_tool_end":
             tool_output_str = str(data.get('output', '')) 
             yield f"Tool **{data['name']}** finished.\nResult: `{tool_output_str}`"
             if full_response:
                 yield full_response

    if full_response and (not chunk or chunk["event"] != "on_chat_model_stream"):
         yield full_response


# --- Gradio Interface (mostly unchanged) ---
demo = gr.ChatInterface(
    fn=stream_from_agent,
    type="messages",
    title="🤖 AGent template",
    description="Ask about the weather anywhere! Watch as I gather the information step by step.",
    cache_examples=False, 
    save_history=True,
    editable=True,
)

if __name__ == "__main__":
    # Load environment variables
    try:
        openai_api_key = os.getenv("OPENAI_API_KEY")
        if openai_api_key:
            print("OPENAI_API_KEY found.")
        else:
            print("Warning: OPENAI_API_KEY not found in environment variables.")
    demo.launch(debug=True, server_name="0.0.0.0")