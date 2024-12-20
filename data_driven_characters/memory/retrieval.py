import streamlit as st
import os
import pandas as pd
import torch
import json
import random

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

EMBEDDING_MODEL = BGEM3FlagModel(
                model_name_or_path="BAAI/bge-m3",
                #TODO 배포시 device 세팅 변경하기!
                device="mps",
                use_fp16=True,
            )


INFO_MAP = {
    1: "biography",
    2: "paraphrasis_1",
    3: "paraphrasis_2",
}

with open("/Users/ridealist/Desktop/data-driven-characters/data/characters.json", 'r') as json_file:
    character_info = json.load(json_file)

# with open('/Users/ridealist/Desktop/data-driven-characters/eval_data/turingcat_test_in_domain.json', 'r') as json_file:
#     in_domain_data = json.load(json_file)
# json_file.close() 

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
    k: int
    black_list: str = "none"

    def _query(self, query):
        with Elasticsearch(
            hosts=elasticsearch_host_url,
            api_key=elasticsearch_api_key, 
        ) as client:
            batch_embeddings = EMBEDDING_MODEL.encode(
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
        return response['hits']['hits']

    def _get_relevant_documents(self, query: str) -> str:
        documents = []
        dialogues = self._query(query)
        for dialogue_dict in dialogues:
            dialogue_texts = []
            # print(f'BLACKLIST: {self.black_list}')
            # print(f'DIALOGUE_DICT: {dialogue_dict}')
            if self.black_list is not None and self.black_list in dialogue_dict['fields']['dialogue_id']:
                continue
            for idx, utterance in enumerate(dialogue_dict['fields']['dialogues']):
                utter_character_id = dialogue_dict['fields']['speakers'][idx]
                try:
                    character_name = character_info[utter_character_id]['character_name']
                except:
                    character_name = '(Unknown)'
                character_name_firstCap = character_name.title()
                dialogue_texts.append(f'{character_name_firstCap}: {utterance}')
            dialogue_str = "\n".join(dialogue_texts)
            documents.append(Document(page_content=dialogue_str, metadata={"conversation_id": dialogue_dict['fields']['dialogue_id'][0]}))
            if len(documents) == self.k:
                break
        return documents
        # return "\n---\n".join(documents) + "\n---\n"


class CornellSearchRetriever(BaseRetriever, BaseModel):
    character_id: str
    k: int
    black_list: str = "none"

    def _query(self, query):
        with Elasticsearch(
            hosts=elasticsearch_host_url,
            api_key=elasticsearch_api_key, 
        ) as client:
            batch_embeddings = EMBEDDING_MODEL.encode(query, return_dense=True, return_sparse=False)

            dense_embeddings = batch_embeddings["dense_vecs"].tolist()
            response = client.search(
                index="search-cornell",
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
                                                "unique_speakers": self.character_id  # 검색에 제한할 speaker
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
        return response['hits']['hits']

    def _get_relevant_documents(self, query: str) -> str:
        documents = []
        dialogues = self._query(query)
        for dialogue_dict in dialogues:
            dialogue_texts = []
            # print(f'BLACKLIST: {self.black_list}')
            # print(f'DIALOGUE_DICT: {dialogue_dict}')
            if self.black_list is not None and self.black_list in dialogue_dict['fields']['dialogue_id']:
                continue
            for idx, utterance in enumerate(dialogue_dict['fields']['dialogues']):
                utter_character_id = dialogue_dict['fields']['speakers'][idx]
                try:
                    character_name = character_info[utter_character_id]['character_name']
                except:
                    character_name = '(Unknown)'
                character_name_firstCap = character_name.title()
                dialogue_texts.append(f'{character_name_firstCap}: {utterance}')
            dialogue_str = "\n".join(dialogue_texts)
            documents.append(Document(page_content=dialogue_str, metadata={"conversation_id": dialogue_dict['fields']['dialogue_id'][0]}))
            if len(documents) == self.k:
                break
        return documents
        # return "\n---\n".join(documents) + "\n---\n"


class ProdigyBiographySearchRetriever(BaseRetriever, BaseModel):
    character_id: str
    k: int
    info_map: dict = {
        1: "biography",
        2: "paraphrasis_1",
        3: "paraphrasis_2",
    }

    def _query(self, query):
        with Elasticsearch(
            hosts=elasticsearch_host_url,
            api_key=elasticsearch_api_key, 
        ) as client:
            batch_embeddings = EMBEDDING_MODEL.encode(query, return_dense=True, return_sparse=False)

            dense_embeddings = batch_embeddings["dense_vecs"].tolist()
            response = client.search(
                index="character-prodigy",
                body={
                    "_source": False,
                    "fields": [
                        "character_id",
                        "character_name",
                        # "movie_id",
                        # "movie_name",
                        # "gender",
                        # "mbti",
                        "biography",
                        "paraphrasis_1",
                        "paraphrasis_2",
                    ],
                    "query": {
                        "bool": {  # Use a bool query to combine conditions
                            "must": [
                                # {
                                #     "term": {
                                #         "character_id": condition_user,  # 검색에 제한할 speaker
                                #     }
                                # },
                                # dense 검색을 사용할거라면 아래 주석 활성화.
                                {
                                
                                    "script_score": {
                                        "query": {
                                            "term": {
                                                "character_id": self.character_id,  # 검색에 제한할 speaker
                                            }
                                        },
                                        "script": {
                                            "source": "dotProduct(params.queryDenseVector, 'dense_biography') + dotProduct(params.queryDenseVector, 'dense_paraphrasis_1') + dotProduct(params.queryDenseVector, 'dense_paraphrasis_2')",
                                            "params": {
                                                "queryDenseVector": dense_embeddings
                                            }
                                        }
                                    }
                                },

                                # term matching을 사용할거람  아래 주석 활성화
                                # {
                                #     "multi_match": {
                                #         "query": query,
                                #         "fields": ["biography", "paraphrasis_1", "paraphrasis_2"],
                                #         "type": "best_fields"  # Use "best_fields" for highest scoring match
                                #     }
                                # }
                            ]
                        }
                    },
                    "size": 10,
                    "track_total_hits": True
                },
                request_timeout=60
            )
        return response['hits']['hits']

    def _get_relevant_documents(self, query: str) -> str:
        documents = []
        biographies = self._query(query)
        for bio_dict in biographies:
            # print(f'DIALOGUE_DICT: {dialogue_dict}')
            which_idx = random.randint(1,3)
            bio_info = bio_dict['fields'][INFO_MAP[which_idx]]
            for bio_str in bio_info:
                documents.append(Document(page_content=bio_str, metadata={"conversation_id": dialogue_dict['fields']['dialogue_id'][0]}))
            if len(documents) == self.k:
                break
        return documents
        # return "\n".join(documents)
