from langgraph.graph import StateGraph, END
from graph.state import GraphState
from graph.nodes.web_search import web_search
from graph.nodes.generate import generate
from graph.nodes.grade_documents import grade_documents
from graph.nodes.retrieve import retrieve
from graph.consts import RETRIEVE, GRADE_DOCUMENTS, GENERATE, WEBSEARCH
from graph.chains.hallucinations_grader import hallucination_grader
from graph.chains.answer_grader import answer_grader
from graph.chains.query_router import router_chain, RouteQuery

def route_question(state: GraphState) -> str:
    print("---ROUTE QUESTION---")
    question = state["question"]
    source: RouteQuery = router_chain.invoke({"question": question})
    if source.datasource == WEBSEARCH:
        print("---ROUTE QUESTION TO WEB SEARCH---")
        return WEBSEARCH
    elif source.datasource == "vectorstore":
        print("---ROUTE QUESTION TO RAG---")
        return RETRIEVE

def decide_search_generate(state: GraphState):
    print("---ASSESS GRADED DOCUMENTS---")

    if state["web_search"]:
        print(
            "---DECISION: NOT ALL DOCUMENTS ARE NOT RELEVANT TO QUESTION, INCLUDE WEB SEARCH---"
        )
        return WEBSEARCH
    else:
        print("---DECISION: GENERATE---")
        return GENERATE

def grade_generation_grounded_in_documents_and_question(state: GraphState):
    print("--CHECKING HALLUCINATIONS--")

    question = state['question']
    documents = state['documents']
    generation = state['generation']

    score = hallucination_grader.invoke(
        {"documents": documents, "generation": generation}
    )

    if hallucination_grade := score.binary_score:
        print("---DECISION: GENERATION IS GROUNDED IN DOCUMENTS---")
        print("---GRADE GENERATION vs QUESTION---")
        score = answer_grader.invoke({"question": question, "generation": generation})
        if answer_grade := score.binary_score:
            print("---DECISION: GENERATION ADDRESSES QUESTION---")
            return "useful"
        else:
            print("---DECISION: GENERATION DOES NOT ADDRESS QUESTION---")
            return "not useful"
    else:
        print("---DECISION: GENERATION IS NOT GROUNDED IN DOCUMENTS, RE-TRY---")
        return "not supported"
    

workflow = StateGraph(GraphState)
workflow.add_node(RETRIEVE, retrieve)
workflow.add_node(GENERATE, generate)
workflow.add_node(WEBSEARCH, web_search)
workflow.add_node(GRADE_DOCUMENTS, grade_documents)

workflow.set_conditional_entry_point(route_question)
workflow.add_edge(RETRIEVE, GRADE_DOCUMENTS)
workflow.add_conditional_edges(GRADE_DOCUMENTS, decide_search_generate)
workflow.add_edge(WEBSEARCH, GENERATE)
workflow.add_conditional_edges(GENERATE, grade_generation_grounded_in_documents_and_question,
                               path_map={
                                   "useful":END,
                                   "not useful":WEBSEARCH,
                                   "not supported":GENERATE
                               })
workflow.add_edge(GENERATE, END)


app = workflow.compile()
print(app.get_graph().draw_mermaid())
