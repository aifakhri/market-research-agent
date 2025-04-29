from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode, tools_condition

from models import AgentState, llm
from tools import load_tools
from ._edges import GradeEdge
from ._nodes import (
    AgentNode,
    RewriteNode,
    GenerateNode,
) 


class AgenticGraph:
    def __init__(self):
        self.llm = llm()
        self.tools = load_tools()
        self.agent_node = AgentNode(llm=self.llm, tools=self.tools)
        self.retrieve_node = ToolNode(self.tools)
        self.rewrite_node = RewriteNode(llm=self.llm)
        self.generate_node = GenerateNode(llm=self.llm)
        self.grade_edge = GradeEdge(llm=self.llm)

    def build(self):
        workflow = StateGraph(AgentState)

        # Add nodes
        workflow.add_node("agent", self.agent_node.run)
        workflow.add_node("retrieve", self.retrieve_node)
        workflow.add_node("rewrite", self.rewrite_node.run)
        workflow.add_node("generate", self.generate_node.run)

        # Add nodes
        # NOTE: Conditional node
        workflow.add_edge(START, "agent")
        workflow.add_conditional_edges(
            "agent",
            tools_condition,
            {
                "tools": "retrieve",
                END: END
            }
        ) 

        workflow.add_conditional_edges(
            "retrieve",
            self.grade_edge.run
        )

        workflow.add_edge("generate", END)
        workflow.add_edge("rewrite", "agent")

        compiled_graph = workflow.compile()

        return compiled_graph