# Market Research Agent

Current development:
- Still following Langgraph rag agent tutorial as is, no modification
- Refactoring all of the agent codes into different separation of concern:
    - Agent tools have its own modules
    - Agent graph and its instantiation would be separated
    - Database module is no longer needed because the vector store doesn't fit the context of tools
- There would be two different branch:
    - with Qdrant docker
    - with Qdrant cloud (the Qdrant local would not be developed further)

# This branch is for cloud development