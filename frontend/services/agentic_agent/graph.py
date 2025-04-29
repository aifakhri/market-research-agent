from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode, tools_condition


from models import AgentState, llm
from tools import load_tools
from ._nodes import (
    AgentNode,
    RewriteNode,
    GenerateNode,
    GradeDocsNode
) 


class AgenticGraph:


    def __init__(self):
        self.tools = load_tools()
        self.llm = llm()
        self.llm_with_tools = self.llm.bind_tools(self.tools) 
        self.agent_node = AgentNode()
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