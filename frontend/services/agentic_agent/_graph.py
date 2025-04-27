from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode, tools_condition


from models import AgentState
from ._nodes import (
    AgentNode,
    RetrieveNode,
    RewriteNode,
    GenerateNode,
    GradeDocsNode
) 


state = AgentState()


class AgenticGraph:
    def __init__(self, retriever_tools):
        self.agent_node = AgentNode()
        self.retriever_node = RetrieveNode(retriever_tools)
        self.grade_node = GradeDocsNode()
        self.rewrite_node = RewriteNode()
        self.generate_node = GenerateNode()

    def build(self):
        graph = StateGraph(AgentState)

        # Add nodes
        graph.add_node("agent", self.agent_node.run)
        graph.add_node("retrieve", self.retrieve_node.run)
        graph.add_node("grade", self.grade_node.run)
        graph.add_node("rewrite", self.rewrite_node.run)
        graph.add_node("generate", self.generate_node.run)

        # Add nodes
        # NOTE: Conditional node
        

        return graph.compile()