import streamlit as st

from services.agentic_agent.controller import Chatbot


@st.cache_resource
def load_chatbot():
    return Chatbot()

# Initialize
st.set_page_config(page_title="Agentic RAG Chatbot", page_icon="🤖")
st.title("Agentic RAG Chatbot")

bot = load_chatbot()

# Set up session state
if "messages" not in st.session_state:
    st.session_state.messages = []

user_input = st.chat_input("Ask me anything about the documents...")

if user_input:
    # Let chatbot handle interaction
    updated_messages = bot.chat(user_input)

    # Save updated conversation
    st.session_state.messages = updated_messages

# isplay conversation
for msg in st.session_state.messages:
    if msg.type == "human":
        st.chat_message("user").markdown(msg.content)
    else:
        st.chat_message("assistant").markdown(msg.content)