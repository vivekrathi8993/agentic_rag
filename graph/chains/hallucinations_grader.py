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

class GradeHallucinations(BaseModel):
    """Binary score for hallucination present in generation answer."""

    binary_score: bool = Field(
        description="Answer is grounded in the facts, 'yes' or 'no'"
    )

structured_llm_grader = llm.with_structured_output(GradeHallucinations)

system = """You are a grader assessing whether an LLM generation is grounded in / supported by a set of retrieved facts. \n 
     Give a binary score 'yes' or 'no'. 'Yes' means that the answer is grounded in / supported by the set of facts."""
hallucination_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "Set of facts: \n\n {documents} \n\n LLM generation: {generation}"),
    ]
)

hallucination_grader: RunnableSequence = hallucination_prompt | structured_llm_grader