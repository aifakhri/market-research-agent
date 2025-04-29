from langchain import hub
from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import StrOutputParser

from langfuse.decorators import observe

from langsmith import traceable

from models import AgentState



class AgentNode:
    def __init__(self, llm, tools):
        self.llm_with_tool = llm.bind_tools(tools)

    @observe()
    def run(self, state: AgentState) -> AgentState:
        """
        """

        print("---CALL AGENT---")

        # Get the message from the state
        messages = state["messages"]
        print(f"Agent message: {messages}")

        # Get response from LLM
        response = self.llm_with_tool.invoke(messages)

        # Store the response in the messages
        return {"messages": [response]}

class RewriteNode:
    def __init__(self, llm):
        self.llm = llm

    @observe()
    def run(self, state: AgentState) -> AgentState:
        print("---TRANSFORM QUERY---")
        messages = state["messages"]
        question = messages[0].content

        msg = [
            HumanMessage(
                content=f"""\n
                    Look at the input and try to reason about the underying semantinc intent / meaning \n
                    Here is the initial question:
                    \n ------ \n
                    {question}
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

    @observe()
    def run(self, state: AgentState) -> AgentState:
        print("---GENERATE---")
        messages = state["messages"]
        question = messages[0].content
        last_message = messages[-1]

        docs = last_message.content

        # prompt
        prompt = hub.pull("rlm/rag-prompt")

        rag_chain = prompt | self.llm | StrOutputParser()

        response = rag_chain.invoke({"context": docs, "question": question})

        return {"messages": [response]}