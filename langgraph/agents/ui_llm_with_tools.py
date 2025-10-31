import streamlit as st
from dotenv import load_dotenv
import time

from langchain_community.tools import DuckDuckGoSearchRun, WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain.tools import tool
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.prebuilt import ToolNode

# -----------------------------
# Setup
# -----------------------------
load_dotenv()

# Initialize tools
search = DuckDuckGoSearchRun()
wiki = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())


@tool
def search_web(query: str) -> str:
    """Search the web for recent information."""
    return search.run(query)


@tool
def get_wiki(topic: str) -> str:
    """Get detailed factual info from Wikipedia."""
    return wiki.run(topic)


tools = [search_web, get_wiki]

# Initialize Groq LLM and bind tools
llm = ChatGroq(model="openai/gpt-oss-20B", temperature=0.2).bind_tools(tools)

# System prompt
system_prompt = SystemMessage(
    content=(
        "You are a factual, helpful assistant. "
        "Use the web search or Wikipedia tools as needed. "
        "VERY VERY IMPORTANT: Do NOT call any tool more than **4 times** for a single question. "
        "After using tools, summarize and answer clearly. "
        "Whenever you include facts from tools, provide citations as markdown links. "
        "Format: [source name](URL)."
    )
)

###################### Define Nodes ######################


def llm_node(state: MessagesState):
    """LLM node: generates next message or calls tool."""
    response = llm.invoke([system_prompt] + state["messages"])
    return {"messages": [response]}


tool_node = ToolNode(tools)

###################### Build the Graph ######################
graph = StateGraph(MessagesState)
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

assistant_graph = graph.compile()

# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="LangGraph Groq Agent", page_icon="🤖")
st.title("🤖 LangGraph Agent (Groq + Tools)")
st.caption("Ask anything — the agent can search the web or fetch from Wikipedia.")

# Initialize chat history
if "history" not in st.session_state:
    st.session_state.history = []

# Display previous chat
for msg in st.session_state.history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# User input
user_input = st.chat_input("Ask me something...")

if user_input:
    # Append user query
    st.session_state.history.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Prepare messages for graph
    messages = [system_prompt]
    for m in st.session_state.history:
        if m["role"] == "user":
            messages.append(HumanMessage(content=m["content"]))
        elif m["role"] == "assistant":
            messages.append(AIMessage(content=m["content"]))

    # Create assistant container
    with st.chat_message("assistant"):
        placeholder = st.empty()
        full_response = ""
        shown_tools = set()

        # Run through the graph stream
        for event in assistant_graph.stream({"messages": messages}, stream_mode="values"):
            messages_list = event["messages"]
            last_message = messages_list[-1]

            # When AI calls a tool
            if isinstance(last_message, AIMessage) and getattr(last_message, "tool_calls", None):
                for tool_call in last_message.tool_calls:
                    tool_name = tool_call.get("name")
                    args = tool_call.get("args", {})
                    query = args.get("query") or args.get("topic") or "N/A"
                    notice = f"🔧 **Using tool:** `{tool_name}`  \n Query: *{query}*\n\n"
                    full_response += notice
                    placeholder.markdown(full_response)

                continue

            # When final text response arrives
            if isinstance(last_message, AIMessage) and not getattr(last_message, "tool_calls", None):
                for chunk in llm.stream([system_prompt] + messages_list):
                    if chunk.content:
                        full_response += chunk.content
                        placeholder.markdown(full_response + "▌")
                        time.sleep(0.01)
                placeholder.markdown(full_response)

        # Save assistant reply
        st.session_state.history.append(
            {"role": "assistant", "content": full_response})
