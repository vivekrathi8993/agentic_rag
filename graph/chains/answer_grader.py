from langchain_ibm.chat_models import ChatWatsonx
from ibm_watsonx_ai.metanames import GenTextParamsMetaNames
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from langchain_core.runnables import RunnableSequence
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

class GradeAnswer(BaseModel):

    binary_score: bool = Field(
        description="Answer addresses the question, 'yes' or 'no'"
    )


structured_llm_grader = llm.with_structured_output(GradeAnswer)

system = """You are a grader assessing whether an ai generated answer addresses / resolves a question \n 
     Give a binary score 'yes' or 'no'. Yes' means that the ai generated answer resolves the question."""
answer_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "User question: \n\n {question} \n\n Ai generated answer: {generation}"),
    ]
)

answer_grader: RunnableSequence = answer_prompt | structured_llm_grader
