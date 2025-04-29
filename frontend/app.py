import os
import requests
import streamlit as st

from dotenv import load_dotenv

load_dotenv()


CHAT_API_URL = os.environ.get("CHAT_API_URL")

st.set_page_config(page_title="Agentic RAG Chatbot")
st.title("Agentic RAG Chatbot")

# Set up session state
if "messages" not in st.session_state:
    st.session_state.messages = []


for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


if prompt := st.chat_input("Ask me anything about the documents..."):
    with st.chat_message("user"):
        st.markdown(prompt)

    # Append new prompt (message) from user
    st.session_state.messages.append({
        "role": "user",
        "content": prompt
    })

    try:
        with st.chat_message("assistant"):
            response = requests.post(CHAT_API_URL, json={"question": prompt})
            response.raise_for_status()

            ai_message = response.json()["answer"]

            st.markdown(ai_message)

        st.session_state.messages.append(
            {
                "role": "asistant",
                "content": ai_message
            }
        )

    except Exception as e:
        st.error(f"Something went wrong: {str(e)}")