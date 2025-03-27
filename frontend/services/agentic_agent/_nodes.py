from langchain import hub
from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import StrOutputParser

from langsmith import traceable

from models import State, llm
from ._tools import retriever


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
        HumanMessage(
            content=f""" \n 
                Look at the input and try to reason about the underlying semantic intent / meaning. \n 
                Here is the initial question:
                \n ------- \n
                {question} 
                \n ------- \n
                Formulate an improved question: """,
        )
    ]

    model = llm()
    response = model.invoke(human_msg)
    return {"messages": [response]}

@traceable
def generate(state: State):
    """Generate answers
    """
    print("---GENERATE---")

    messages = state["messages"]
    question = messages[0].content
    last_message = messages[-1]

    docs = last_message.content

    prompt = hub.pull("rlm/rag-prompt")

    model = llm()

    rag_chain = prompt | model | StrOutputParser()

    response = rag_chain.invoke({"context": docs, "question": question})
    return {"messages": [response]}