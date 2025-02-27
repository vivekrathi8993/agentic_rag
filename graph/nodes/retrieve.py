from ingestion import retriever
from graph.state import GraphState
from typing import Dict, Any

def retrieve(state : GraphState) -> Dict[str, Any]:
    print("---RETRIEVE---")
    question = state["question"]
    documents = retriever.invoke(question)[:2]
    return {"documents": documents, "question": question}
