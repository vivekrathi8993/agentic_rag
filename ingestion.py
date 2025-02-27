from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_chroma import Chroma
from langchain_ibm import ChatWatsonx
from ibm_watsonx_ai.metanames import GenTextParamsMetaNames
from ibm_watsonx_ai.metanames import EmbedTextParamsMetaNames as EmbedParams
from ibm_watsonx_ai.foundation_models.utils.enums import EmbeddingTypes
from ibm_watsonx_ai.foundation_models import Embeddings
from ibm_watsonx_ai import Credentials
from ibm_watsonx_ai import APIClient

import os

load_dotenv()

#PARAMS
USE_CLOUD = USE_CLOUD = os.getenv("USE_CLOUD").lower() in ('true', '1', 't')

if USE_CLOUD:
    #print("Using watsonx SaaS....")
    WATSONX_API_KEY = os.getenv("WATSONX_API_KEY_CLOUD")
    WATSONX_PROJECT_ID = os.getenv("WATSONX_PROJECT_ID_CLOUD")
    WATSONX_URL = os.getenv("WATSONX_URL_CLOUD")
else:
    #print("Using self-hosted watsonx env...")
    WATSONX_API_KEY = os.getenv("WATSONX_API_KEY")
    WATSONX_PROJECT_ID = os.getenv("WATSONX_PROJECT_ID")
    WATSONX_URL = os.getenv("WATSONX_URL")

credentials = Credentials(
    url=WATSONX_URL,
    api_key=WATSONX_API_KEY,
)
# List available embedding models
api_client = APIClient(credentials=credentials, project_id=WATSONX_PROJECT_ID)
#print(api_client.foundation_models.EmbeddingModels.show())

# Set the truncate_input_tokens to a value that is equal to or less than the maximum allowed tokens for the embedding model that you are using. If you don't specify this value and the input has more tokens than the model can process, an error is generated.

embed_params = {
EmbedParams.TRUNCATE_INPUT_TOKENS: 128,
 EmbedParams.RETURN_OPTIONS: {
 'input_text': True
 }
}

embedding = Embeddings(
 model_id='sentence-transformers/all-minilm-l12-v2',
 credentials=credentials,
 params=embed_params,
 project_id=WATSONX_PROJECT_ID,
 space_id=None,
 verify=False
)


urls = [
    "https://lilianweng.github.io/posts/2023-06-23-agent/",
    "https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/",
    "https://lilianweng.github.io/posts/2023-10-25-adv-attack-llm/",
]

if os.path.exists('./.chroma'):

    #print("Retrieving vectore store...")
    retriever = Chroma(
        collection_name="rag-chroma",
        persist_directory="./.chroma",
        embedding_function=embedding
    ).as_retriever()

else:
    print("Loading full data ... ")
    docs = [WebBaseLoader(url).load() for url in urls]
    docs_list = [item for sublist in docs for item in sublist]

    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size = 1024, chunk_overlap=256
    )

    doc_splits = text_splitter.split_documents(docs_list)

    vectorstore = Chroma.from_documents(
        documents=doc_splits,
        persist_directory='./.chroma',
        collection_name="rag-chroma",
        embedding=embedding
    )

    retriever = Chroma(
        collection_name="rag-chroma",
        persist_directory="./.chroma",
        embedding_function=embedding
    ).as_retriever()    

#print(retriever.invoke("whats the best use of llms"))