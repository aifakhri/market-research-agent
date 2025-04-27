from langchain import hub
from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import StrOutputParser

from langgraph.prebuilt import ToolNode

from langsmith import traceable

from models import AgentState



class AgentNode:
    def __init__(self, llm_with_tools):
        self.llm = llm_with_tools

    def run(self, state: AgentState) -> AgentState:
        """
        """

        # Get the message from the state
        messages = state["messages"]

        # Get response from LLM
        response = self.llm.invoke(messages)

        # Store the response in the messages
        return {"message": [response]}

class RewriteNode:
    def __init__(self, llm):
        self.llm = llm

    def run(self, state: AgentState) -> AgentState:
        print("--TRANSFORM QUERY--")
        messages = state["messages"]
        query = messages[0].content

        msg = [
            HumanMessage(
                content=f"""\n
                    Look at the input and try to reason about the underying semantinc intent / meaning \n
                    Here is the initial question:
                    \n ------ \n
                    {query}
                    \n ------ \n
                    Formulate an improved question:
                """
            )
        ]

        # Grader
        response = self.llm.invoke(msg)
        return {"messages": [response]}

class GenerateNode:
    def __init__(self, llm):
        self.llm = llm

    def run(self, state: AgentState) -> AgentState:
        print("--GENERATE--")
        messages = state["messages"]
        question = messages[0].content
        last_message = messages[-1]

        docs = last_message.content

        # prompt
        prompt = hub.pull("rlm/rag-prompt")

        rag_chain = prompt | self.llm | StrOutputParser()

        response = rag_chain.invoke({"context": docs, "question": question})

        return {"messages": [response]}