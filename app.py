import os
import gradio as gr
from gradio import ChatMessage
import requests
from typing import Dict, List
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent

# Weather and location tools
@tool
def get_lat_lng(location_description: str) -> dict[str, float]:
    """Get the latitude and longitude of a location."""
    return {"lat": 51.1, "lng": -0.1}  # London coordinates as dummy response

@tool
def get_weather(lat: float, lng: float) -> dict[str, str]:
    """Get the weather at a location."""
    return {"temperature": "21°C", "description": "Sunny"}  # Dummy response
        

def stream_from_agent(message: str, history: List[Dict[str, str]]) -> gr.ChatMessage:
    """Process messages through the LangChain agent with visible reasoning."""
    
    # Initialize the agent
    llm = ChatOpenAI(temperature=0, model="gpt-4")
    memory = MemorySaver()
    tools = [get_lat_lng, get_weather]
    agent_executor = create_react_agent(llm, tools, checkpointer=memory)
    
    # Add message to history
    past_messages = [HumanMessage(content=message)]
    for h in history:
        if h["role"] == "user":
            past_messages.append(HumanMessage(content=h["content"]))
    
    messages_to_display = []
    final_response = None
    
    for chunk in agent_executor.stream(
        {"messages": past_messages},
        config={"configurable": {"thread_id": "abc123"}}
    ):
        # Handle agent's actions and tool usage
        if chunk.get("agent"):
            for msg in chunk["agent"]["messages"]:
                if msg.content:
                    final_response = msg.content
                
                # Handle tool calls
                for tool_call in msg.tool_calls:
                    tool_message = ChatMessage(
                        content=f"Parameters: {tool_call['args']}",
                        metadata={
                            "title": f"🛠️ Using {tool_call['name']}",
                            "id": tool_call["id"],
                            "status": "pending",
                        }
                    )
                    messages_to_display.append(tool_message)
                    yield messages_to_display
                    tool_message.metadata["status"] = "done"
                    
        # Handle tool responses
        if chunk.get("tools"):
            for tool_response in chunk["tools"]["messages"]:
                # Find the corresponding tool message
                for msg in messages_to_display:
                    if msg.metadata.get("id") == tool_response.tool_call_id:
                        msg.content += f"\nResult: {tool_response.content}"
                        yield messages_to_display
    
    # Add the final response as a regular message
    if final_response:
        messages_to_display.append(ChatMessage(content=final_response))
        yield messages_to_display

# Create the Gradio interface
demo = gr.ChatInterface(
    fn=stream_from_agent,
    type="messages",
    title="🌤️ Weather Assistant",
    description="Ask about the weather anywhere! Watch as I gather the information step by step.",
    examples=[
        "What's the weather like in Tokyo?",
        "Is it sunny in Paris right now?",
        "Should I bring an umbrella in New York today?"
    ],
    example_icons=["https://cdn3.iconfinder.com/data/icons/landmark-outline/432/japan_tower_tokyo_landmark_travel_architecture_tourism_view-256.png", 
                   "https://cdn2.iconfinder.com/data/icons/city-building-1/200/ArcdeTriomphe-256.png",
                   "https://cdn2.iconfinder.com/data/icons/city-icons-for-offscreen-magazine/80/new-york-256.png"
                  ],
    save_history=True,
    editable=True

)

if __name__ == "__main__":
    # Load environment variables
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass

    demo.launch(debug=True)