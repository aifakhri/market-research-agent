from langchain_core.messages import HumanMessage

from langfuse.callback import CallbackHandler

from models import AgentState
from services.agentic_agent._graph import AgenticGraph


langfuse_handler = CallbackHandler()

class Chatbot:
    def __init__(self):
        self.state = AgentState()
        self.graph = AgenticGraph().build()

    def chat(self, user_input: str):
        user_message = {
            "messages": [
                ("user", user_input)
            ]
        }

        # state = AgentState(messages=[HumanMessage(content=user_input)])
        result = self.graph.invoke(
            user_message,
            config={"callbacks": [langfuse_handler]})
        return result["messages"]