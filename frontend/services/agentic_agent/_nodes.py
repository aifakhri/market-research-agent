from langchain_core.messages import HumanMessage

from models import State, llm
from _tools import retriever


tools = [retriever]


def agent(state: State):
    """Agent invocation
    """

    print("---CALL AGENT---")
    
    messages = state["messages"]
    model = llm()
    model = model.bind_tools(tools)
    response = model.invoke(messages)

    return {"messages": [response]}

def rewrite(state: State):
    """Transform the query to produce a better question
    """

    print("---TRANSFORM QUERY---")

    messages = state["messages"]
    question = messages[0].content

    human_msg = [
        HumanMessage
    ]
    