from dotenv import load_dotenv
from langchain_ibm.chat_models import ChatWatsonx
from ibm_watsonx_ai.metanames import GenTextParamsMetaNames
from langchain.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from typing import Literal
import os

class RouteQuery(BaseModel):
    """Route a user query to the most relevant datasource."""

    datasource: Literal["vectorstore", "websearch"] = Field(
        ...,
        description="Given a user question choose to route it to web search or a vectorstore.",
    )

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
    GenTextParamsMetaNames.TEMPERATURE:0,
    GenTextParamsMetaNames.MAX_NEW_TOKENS:10,
    GenTextParamsMetaNames.MAX_NEW_TOKENS:2
}

model = "ibm/granite-3-8b-instruct"

llm = ChatWatsonx(model_id=model, apikey=WATSONX_API_KEY, project_id=WATSONX_PROJECT_ID, url=WATSONX_URL)

llm_structured_output = llm.with_structured_output(RouteQuery)

system = """You are an expert at routing a user question to a vectorstore or web search.
The vectorstore contains documents related to agents, prompt engineering, and adversarial attacks.
Use the vectorstore for questions on these topics. For all else, use web-search."""

router_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "{question}")
    ]
)

router_chain = router_prompt | llm_structured_output

#print(router_chain.invoke(input={"question":'how does a f1 car technically be so fast?'}))