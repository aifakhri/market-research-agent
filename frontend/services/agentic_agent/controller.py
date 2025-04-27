from models import AgentState



class Chatbot:
    def __init__(self, graph):
        self.graph = graph
        self.state = AgentState()

    def chat(self, query: str):
        result = self.graph.invoke(self.state)

        return result