import streamlit as st

from services import ChatService


chatbot = ChatService()
chatbot.initializing_graph()


# # NOTE: initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input(placeholder="Enter your message"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        latest_messages = [
            {"role": m["role"], "content": m["content"]}
            for m in st.session_state.messages
        ]

        response = st.write_stream(chatbot.stream_graph(messages=latest_messages))
    st.session_state.messages.append({"role": "assistant", "content": response})