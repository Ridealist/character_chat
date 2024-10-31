import streamlit as st
import os

from typing import Any, List, Dict, ClassVar
from langchain.memory import VectorStoreRetrieverMemory
from langchain.schema import Document

from FlagEmbedding import BGEM3FlagModel
from elasticsearch import Elasticsearch

elasticsearch_api_key = st.secrets["elasticsearch_api_key"]
elasticsearch_host_url = st.secrets["elasticsearch_host_url"]
os.environ["ELASTIC_API_KEY"] = elasticsearch_api_key
os.environ["ELASTIC_HOST_URL"] = elasticsearch_host_url


class ConversationVectorStoreRetrieverMemory(VectorStoreRetrieverMemory):
    input_prefix: ClassVar[str]
    output_prefix: ClassVar[str]
    blacklist: ClassVar[List]

    input_prefix = "Human"
    output_prefix = "AI"
    blacklist = []  # keys to ignore

    def _form_documents(
        self, inputs: Dict[str, Any], outputs: Dict[str, str]
    ) -> List[Document]:
        """Format context from this conversation to buffer."""
        # Each document should only include the current turn, not the chat history
        filtered_inputs = {
            k: v
            for k, v in inputs.items()
            if k != self.memory_key and k not in self.blacklist
        }
        texts = []
        for k, v in list(filtered_inputs.items()) + list(outputs.items()):
            if k == "input":
                k = self.input_prefix
            elif k == "response":
                k = self.output_prefix
            texts.append(f"{k}: {v}")
        page_content = "\n".join(texts)
        return [Document(page_content=page_content)]


class ElasticSearchRetrievalMemory:

    def __init__(self):
        self.model = BGEM3FlagModel("BAAI/bge-m3")

    def retrieve(self, query):
        with Elasticsearch(
            hosts=elasticsearch_host_url, 
            api_key=elasticsearch_api_key,
            ) as client:
                batch_embeddings = self.model.encode(
                    query, return_dense=True, return_sparse=False
                )
                dense_embeddings = batch_embeddings["dense_vecs"].tolist()
                response = client.search(
                    index=f"search-prodigy",
                    body={
                        "fields": [
                            "dialogue_id",
                            "unique_speakers",
                            "dialogues",
                            "speakers"
                        ],
                        "query": {
                            "script_score": {
                                "query": {
                                    "match_all": {}
                                },
                                "script": {
                                    "source": "dotProduct(params.queryDenseVector, 'dense_embedding')",
                                    "params": {
                                        "queryDenseVector": dense_embeddings
                                    }
                                }
                            }
                        }
                    },
                    request_timeout=60
                )

        return response['hits']['hits'][0]
