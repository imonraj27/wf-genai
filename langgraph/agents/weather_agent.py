# Import necessaries
import requests
from langchain.tools import tool
from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver  # optional but helps
from langgraph.prebuilt import ToolNode
from typing import TypedDict, Annotated, Sequence
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, BaseMessage
from dotenv import load_dotenv

load_dotenv()


###################### Weather fetcher tool ######################
@tool("fetch_weather")
def fetch_weather(location: str) -> dict | None:
    """
    Fetches the current weather information for a given location using the Open-Meteo API.

    Args:
        location (str): The name of the city or location to get weather data for.

    Returns:
        dict | None: JSON containing weather data with a readable weather condition,
        or None if the data could not be retrieved.
    """
    print(f"TOOL IS CALLED - fetch_weather({location})")

    try:
        # Step 1: Get latitude and longitude for the city
        geo_url = f"https://geocoding-api.open-meteo.com/v1/search?name={location}&count=1"
        geo_res = requests.get(geo_url, timeout=10)
        geo_data = geo_res.json()

        if "results" not in geo_data or not geo_data["results"]:
            return None

        lat = geo_data["results"][0]["latitude"]
        lon = geo_data["results"][0]["longitude"]
        city = geo_data["results"][0]["name"]
        country = geo_data["results"][0].get("country", "")

        # Step 2: Get current weather for that coordinate
        weather_url = (
            f"https://api.open-meteo.com/v1/forecast"
            f"?latitude={lat}&longitude={lon}&current_weather=true"
        )
        weather_res = requests.get(weather_url, timeout=10)
        weather_data = weather_res.json()

        current = weather_data.get("current_weather", {})
        if not current:
            return None

        # Replace numeric weather code with description
        weather_map = {
            0: "clear sky ☀️",
            1: "mainly clear 🌤️",
            2: "partly cloudy ⛅",
            3: "overcast ☁️",
            45: "foggy 🌫️",
            48: "depositing rime fog 🌫️",
            51: "light drizzle 🌦️",
            61: "rainy 🌧️",
            71: "snowfall ❄️",
            95: "thunderstorm ⛈️"
        }

        code = current.get("weathercode", -1)
        current["weather_description"] = weather_map.get(
            code, f"unknown (code {code})")

        # Add contextual info for clarity
        result = {
            "city": city,
            "country": country,
            "latitude": lat,
            "longitude": lon,
            "current_weather": current
        }

        return result

    except Exception as e:
        print("Error fetching weather:", e)
        return None


tools = [fetch_weather]

###################### Setting up the LLM ######################
system_prompt = SystemMessage(content="""
    You are a helpful weather assistant.

    Your goal is to provide accurate and concise weather information
    for a given location, using the available tools when necessary.

    You have access to a tool called `fetch_weather(location: str)` 
    which can fetch the current weather for a given city or place.

    When the user asks about the weather:
    - Identify the location mentioned (if any).
    - Use the `fetch_weather` tool to get weather data.
    - If the tool fails or returns nothing, politely inform the user that
      the data is currently unavailable.
    - Format your final message in a friendly and natural way.
    - Use emojis to show the feel of the weather.

    If the user’s request is unrelated to weather, respond that
    you can only assist with weather-related questions.
    """
                              )

llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash", temperature=0
).bind_tools(tools=tools)


###################### Define Graph State ######################
class WeatherState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]


###################### Define Nodes ######################
def llm_node(state: WeatherState):
    """LLM node: generates next message or calls tool."""
    response = llm.invoke([system_prompt] + state["messages"])
    return {"messages": [response]}


tool_node = ToolNode(tools)

###################### Build the Graph ######################
graph = StateGraph(WeatherState)
graph.add_node("llm", llm_node)
graph.add_node("tools", tool_node)

graph.add_conditional_edges(
    "llm",
    lambda state: (
        "tools" if state["messages"][-1].tool_calls else END
    ),
)

graph.add_edge("tools", "llm")
graph.add_edge(START, "llm")

weather_graph = graph.compile()

###################### Run Example ######################
if __name__ == "__main__":
    question = input("Ask about the weather: ")
    result = weather_graph.invoke(WeatherState(
        messages=[HumanMessage(content=question)]))
    final_message = result["messages"][-1].content
    print("\n🌤️ Final Answer:", final_message)
