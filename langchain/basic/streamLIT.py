import streamlit as st
from langchain_ollama.chat_models import ChatOllama
from langchain_google_genai.chat_models import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage
from dotenv import load_dotenv

load_dotenv()

# ----------------- CONFIG -----------------
st.set_page_config(page_title="Chat with Aakash", page_icon="💬")

st.title("💬 Chat with Aakash — Your Senior Engineer")
st.caption("Ask technical questions and get helpful, professional answers.")

# ----------------- MODEL SETUP -----------------
# llm = ChatOllama(model="gemma2:2b")
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.2)


prompt = ChatPromptTemplate([
    (
        "system",
        """
        You are Aakash, a Senior Software Engineer and team lead.  
        A group of junior engineers reports to you and often asks technical questions related to their work.

        Your responsibilities:
        - Always start your response with: "Hi [JUNIOR_NAME],"
        - Provide a short, clear, and professional answer.
        - Be kind, patient, and encouraging — you’re mentoring, not just instructing.
        - Include a relevant code snippet whenever it helps clarify your explanation.
        - Avoid open-ended questions or requests for more information — your answer should be self-contained.
        - Keep answers concise (a few sentences plus optional code).
        - Never break character as Aakash.

        Your goal: Help juniors understand the solution efficiently and confidently.
        """
    ),
    MessagesPlaceholder(variable_name="conversation"),
    ("human", "Hi Aakash, {QUESTION} - From {JUNIOR_NAME}")
])

parser = StrOutputParser()
chain = prompt | llm | parser

# ----------------- SESSION STATE -----------------
if "messages" not in st.session_state:
    st.session_state.messages = []

if "llm_messages" not in st.session_state:
    st.session_state.llm_messages = []

junior_name = st.text_input("👤 Enter your name", key="junior_name")

# ----------------- CHAT UI -----------------
st.write("💡 Ask Aakash anything about your engineering work!")

for msg in st.session_state.messages:
    if msg["role"] == "user":
        with st.chat_message("user"):
            st.markdown(f"**{junior_name or 'You'}:** {msg['content']}")
    else:
        with st.chat_message("assistant"):
            st.markdown(f"**Aakash:** {msg['content']}")

if prompt_text := st.chat_input("Type your question for Aakash..."):
    if not junior_name.strip():
        st.warning("Please enter your name above first.")
    else:
        # Display user message
        with st.chat_message("user"):
            st.markdown(f"**{junior_name}:** {prompt_text}")

        st.session_state.messages.append(
            {"role": "user", "content": prompt_text})

        st.session_state.llm_messages.append(HumanMessage(
            content=prompt_text + " - From: " + junior_name))

        # Stream Aakash's response
        with st.chat_message("assistant"):
            try:
                response_placeholder = st.empty()
                full_response = ""

                # stream() yields tokens/chunks incrementally
                for chunk in chain.stream({
                    "conversation": st.session_state.llm_messages,
                    "QUESTION": prompt_text,
                    "JUNIOR_NAME": junior_name
                }):
                    full_response += chunk
                    response_placeholder.markdown(
                        f"**Aakash:** {full_response}▌")

                response_placeholder.markdown(
                    f"**Aakash:** {full_response}")

                st.session_state.messages.append(
                    {"role": "assistant", "content": full_response})

                st.session_state.llm_messages.append(
                    AIMessage(content=full_response))

            except Exception as e:
                st.error(f"⚠️ Error: {e}")
