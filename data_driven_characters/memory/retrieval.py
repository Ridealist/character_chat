import streamlit as st
import os
import pandas as pd
import torch

from typing import Any, List, Dict, ClassVar
from pydantic import BaseModel
from langchain.memory import VectorStoreRetrieverMemory
from langchain.schema import Document
from langchain_core.retrievers import BaseRetriever

from FlagEmbedding import BGEM3FlagModel
from elasticsearch import Elasticsearch

from data_driven_characters.constants import NUM_CONTEXT_MEMORY

elasticsearch_api_key = st.secrets["elasticsearch_api_key"]
elasticsearch_host_url = st.secrets["elasticsearch_host_url"]
os.environ["ELASTIC_API_KEY"] = elasticsearch_api_key
os.environ["ELASTIC_HOST_URL"] = elasticsearch_host_url

torch.set_num_threads(1)

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


class ElasticSearchRetriever(BaseRetriever, BaseModel):
    character_id: str
    characters_info_df: pd.DataFrame
    k: int


    def _query(self, query):
        with Elasticsearch(
            hosts=elasticsearch_host_url,
            api_key=elasticsearch_api_key, 
        ) as client:
            batch_embeddings = BGEM3FlagModel(
                model_name_or_path="BAAI/bge-m3",
                device="mps"
            ).encode(
                query, return_dense=True, return_sparse=False
            )
            dense_embeddings = batch_embeddings["dense_vecs"].tolist()

            response = client.search(
                    index="search-prodigy",
                    body={
                        "_source": False,
                        "fields": [
                            "dialogue_id",
                            "unique_speakers",
                            "dialogues",
                            "speakers",
                        ],
                        "query": {
                            "bool": {  # Use a bool query to combine conditions
                                "must": [
                                    {
                                        "script_score": {
                                            "query": {
                                                "term": {
                                                    "unique_speakers": self.character_id # 검색에 제한할 speaker 
                                                }
                                            },
                                            "script": {
                                                "source": "dotProduct(params.queryDenseVector, 'dense_embedding');",
                                                "params": {
                                                    "queryDenseVector": dense_embeddings
                                                }
                                            }
                                        }
                                    }
                                ]
                            }
                        },
                        "size": 10,
                        "track_total_hits": True
                    },
                    request_timeout=60
                )

        #TODO Add more dialogues to response - by score threshold -> retrieve all related dialogues
        return response['hits']['hits'][:self.k]


    def _get_relevant_documents(self, query: str) -> List[Document]:
        documents = []
        dialogues = self._query(query)
        # print("-"*25 + 'DIALOGUES' + "-"*25)
        # print(dialogues)
        # print("-"*50)
        for dialogue_dict in dialogues:
            dialogue_texts = []
            for idx, utterance in enumerate(dialogue_dict['fields']['dialogues']):
                utter_character_id = dialogue_dict['fields']['speakers'][idx]
                try:
                    character_name = self.characters_info_df[self.characters_info_df['character_id'] == utter_character_id]['character_name'].item()
                except:
                    character_name = 'Unknown'
                character_name_firstCap = character_name[0] + character_name[1:].lower()
                dialogue_texts.append(f'{character_name_firstCap}: {utterance}')
            page_content = "\n".join(dialogue_texts)
            documents.append(Document(page_content=page_content))
        return documents

