from langchain_openai import ChatOpenAI
from langchain_community.tools.tavily_search import TavilySearchResults

from langgraph.graph import StateGraph, START, END

from models import State
from const import LLM_MODEL
from .chat_tools import BasicToolNode



def route_tools(state: State):
    if isinstance(state, list):
        ai_message = state[-1]
    elif messages := state.get("messages", []):
        ai_message = messages[-1]
    else:
        raise ValueError(f"No message found in input state to tool_edge: {state}")

    # Condition to route agent
    if hasattr(ai_message, "tool_calls") and len(ai_message.tool_calls) > 0:
        return "tools"
    return END



class ChatService:
    def __init__(self):
        self.llm = ChatOpenAI(model=LLM_MODEL)
        self._graph_builder = StateGraph(State)
        self._tool_tavily = TavilySearchResults(max_results=2)

    def _chatbot(self, state: State):
        llm_with_tools = self.llm.bind_tools(self._tools)
        return {"messages": [llm_with_tools.invoke(state["messages"])]}

    def _chat_tools(self):
        self._tools = [self._tool_tavily]
        self._tool_node = BasicToolNode(tools=self._tools)

    def initializing_graph(self):
        # NOTE: Initialize tools
        self._chat_tools()

        # NOTE: build nodes
        self._graph_builder.add_node("chatbot", self._chatbot)
        self._graph_builder.add_node("tools", self._tool_node)

        # NOTE: Define the conditdional edges
        self._graph_builder.add_conditional_edges(
            "chatbot",
            route_tools,
            {"tools": "tools", END: END},
        )

        # NOTE: Define the edges
        self._graph_builder.add_edge("tools", "chatbot")
        self._graph_builder.add_edge(START, "chatbot")

        self.graph = self._graph_builder.compile()

    def stream_graph(self, messages):
        for event in self.graph.stream({"messages": messages}):
            for value in event.values():
                yield value["messages"][-1].content