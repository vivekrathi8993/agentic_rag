from langchain_ibm.chat_models import ChatWatsonx
from ibm_watsonx_ai.metanames import GenTextParamsMetaNames
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from dotenv import load_dotenv
import os

load_dotenv()

USE_CLOUD = os.getenv("USE_CLOUD").lower() in ('true', '1', 't')

if USE_CLOUD:
    print("Using watsonx SaaS....")
    WATSONX_API_KEY = os.getenv("WATSONX_API_KEY_CLOUD")
    WATSONX_PROJECT_ID = os.getenv("WATSONX_PROJECT_ID_CLOUD")
    WATSONX_URL = os.getenv("WATSONX_URL_CLOUD")
else:
    print("Using self-hosted watsonx env...")
    WATSONX_API_KEY = os.getenv("WATSONX_API_KEY")
    WATSONX_PROJECT_ID = os.getenv("WATSONX_PROJECT_ID")
    WATSONX_URL = os.getenv("WATSONX_URL")


params = {
    GenTextParamsMetaNames.DECODING_METHOD:"greedy",
    GenTextParamsMetaNames.MAX_NEW_TOKENS:20,
    GenTextParamsMetaNames.MIN_NEW_TOKENS:5,
    GenTextParamsMetaNames.TEMPERATURE:0
    }

model = "meta-llama/llama-3-2-3b-instruct"

llm = ChatWatsonx(model_id=model, apikey=WATSONX_API_KEY, project_id=WATSONX_PROJECT_ID, url=WATSONX_URL)

class GradeDocuments(BaseModel):
    """Binary score for relevance check on retrieved documents."""

    binary_score: str = Field(
        description="Documents are relevant to the question, 'yes' or 'no'"
    )

structured_llm_grader = llm.with_structured_output(GradeDocuments)


system = """You are a grader assessing relevance of a retrieved document to a user question. \n 
    Examine the document carefully. Respond 'yes' only if the document contains relevant information that answers the question otherwise say 'no'.\n
    Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question."""

grade_prompt = ChatPromptTemplate.from_messages(
    [
    ('system',system),
    ("human", "Retrieved document: \n\n {document} \n\n User question: {question}"),
    ]
)

retrieval_grader = grade_prompt | structured_llm_grader