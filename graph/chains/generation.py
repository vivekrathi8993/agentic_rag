from dotenv import load_dotenv
from langchain import hub
from ibm_watsonx_ai.metanames import GenTextParamsMetaNames
from langchain_ibm import ChatWatsonx
from langchain_core.output_parsers import StrOutputParser
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

model = "meta-llama/llama-3-3-70b-instruct"

llm = ChatWatsonx(model_id=model, apikey=WATSONX_API_KEY, project_id=WATSONX_PROJECT_ID, url=WATSONX_URL)

prompt = hub.pull("rlm/rag-prompt")

generation_chain = prompt | llm | StrOutputParser()