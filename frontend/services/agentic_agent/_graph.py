from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode, tools_condition

from models import State
from ._nodes import rewrite, generate, agent
from ._edges import grade_documents
from ._tools import retriever



class Chatbot:
    def __init__(self):
        pass

    def build_worfklow(self):
        self.workflow = StateGraph(State)

        # define tools
        retriever_ = ToolNode([retriever])

        # Define nodes
        self.workflow.add_node("agent", agent)
        self.workflow.add_node("retrieve", retriever_)
        self.workflow.add_node("rewrite", rewrite)
        self.workflow.add_node("generate", generate)

        # Define edges
        self.workflow.add_edge(START, "agent")
        self.workflow.add_conditional_edges(
            "agent",
            tools_condition,
            {
                "tools": "retrieve",
                END: END,
            }
        )

        self.workflow.add_conditional_edges(
            "retrieve",
            grade_documents
        )
    
        self.workflow.add_edge("generate", END)
        self.workflow.add_edge("rewrite", "agent")

        self.graph = self.workflow.compile()

    def stream_graph(self, inputs):
        for s in self.graph.stream({"messages": inputs}):
            print(s)
            if s.get("agent", []):
                yield s["agent"]["messages"][-1].content
            elif s.get("retriever", []):
                yield s["retriever"]["messages"][-1].content
            elif s.get("generate", []):
                yield s["generate"]["messages"][-1]

    def evaluate_graph(self, inputs: dict) -> dict:
        pass