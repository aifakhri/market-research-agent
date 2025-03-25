from langchain_openai import ChatOpenAI

from langgraph.graph import StateGraph, START, END

from models import State
from const import LLM_MODEL


class ChatService:
    def __init__(self):
        self.llm = ChatOpenAI(model=LLM_MODEL)
        self._graph_builder = StateGraph(State)

    def _chatbot(self, state: State):
        return {"messages": [self.llm.invoke(state["messages"])]}

    def initializing_graph(self):
        self._graph_builder.add_node("chatbot", self._chatbot)
        self._graph_builder.add_edge(START, "chatbot")
        self._graph_builder.add_edge("chatbot", END)

        self.graph = self._graph_builder.compile()
        # return graph

    def stream_graph(self, messages):
        for event in self.graph.stream({"messages": messages}):
            for value in event.values():
                yield value["messages"][-1].content