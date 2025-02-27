from typing import Any, Dict

from graph.chains.retrieval_grader import retrieval_grader
from graph.state import GraphState

def grade_documents(state: GraphState) -> Dict[str, Any]:
    """
    Determines whether the retrieved documents are relevant to the question
    If any document is not relevant, we will set a flag to run web search

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): Filtered out irrelevant documents and updated web_search state
    """

    print("---CHECK DOCUMENT RELEVANCE TO QUESTION---")
    documents = state['documents']
    question = state['question']

    web_search = False
    filtered_documents = []
    for d in documents:
        grade_relevance = retrieval_grader.invoke(input = {"document" : d.page_content, "question" : question})
        relevance_score = grade_relevance.binary_score
        
        if relevance_score.lower() == "yes":
            filtered_documents.append(d)
            
        else:
            web_search = True
            continue        
    
    return {"question":question, "documents":filtered_documents, "web_search": web_search}
    
