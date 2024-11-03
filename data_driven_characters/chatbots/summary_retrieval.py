import faiss
from tqdm import tqdm

from langchain.prompts import PromptTemplate
from langchain.chains.conversation.base import ConversationChain
from langchain.chains.retrieval import create_retrieval_chain
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain.memory import (
    ConversationBufferMemory,
    CombinedMemory,
)
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from data_driven_characters.memory import ConversationVectorStoreRetrieverMemory, ElasticSearchRetriever
from data_driven_characters.constants import NUM_CONTEXT_MEMORY

from pprint import pprint

class SummaryRetrievalChatBot:
    def __init__(self, character_definition, documents):
        self.character_definition = character_definition
        self.documents = documents
        self.num_context_memories = 12

        self.chat_history_key = "chat_history"
        self.context_key = "context"
        self.input_key = "input"

        self.chain = self.create_chain(character_definition)

    def create_chain(self, character_definition):
        conv_memory = ConversationBufferMemory(
            memory_key=self.chat_history_key, input_key=self.input_key
        )
        context_memory = ConversationVectorStoreRetrieverMemory(
            retriever=FAISS(
                OpenAIEmbeddings().embed_query,
                faiss.IndexFlatL2(1536),  # Dimensions of the OpenAIEmbeddings
                InMemoryDocstore({}),
                {},
            ).as_retriever(search_kwargs=dict(k=self.num_context_memories)),
            memory_key=self.context_key,
            output_prefix=character_definition.name,
            blacklist=[self.chat_history_key],
        )
        # add the documents to the context memory
        for i, summary in tqdm(enumerate(self.documents)):
            context_memory.save_context(inputs={}, outputs={f"[{i}]": summary})

        # Combined
        memory = CombinedMemory(memories=[conv_memory, context_memory])
        prompt = PromptTemplate.from_template(
            f"""Your name is {character_definition.name}.
Here is how you describe yourself:
---
{character_definition.long_description}
---

You will have a conversation with a Human, and you will engage in a dialogue with them.
You will exaggerate your personality, interests, desires, emotions, and other traits.
You will stay in character as {character_definition.name} throughout the conversation, even if the Human asks you questions that you don't know the answer to.
You will not break character as {character_definition.name}.

You are {character_definition.name} in the following story snippets, which describe events in your life.
---
{{{self.context_key}}}
---

Current conversation:
---
{character_definition.name}: {character_definition.greeting}
{{{self.chat_history_key}}}
---

Human: {{{self.input_key}}}
{character_definition.name}:"""
        )
        GPT4o = ChatOpenAI(model='gpt-4o')
        # GPT3 = ChatOpenAI(model="gpt-3.5-turbo")
        chatbot = ConversationChain(
            llm=GPT4o, verbose=True, memory=memory, prompt=prompt
        )
        return chatbot

    def greet(self):
        return self.character_definition.greeting

    def step(self, input):
        return self.chain.run(input=input)


class SummaryRetrievalChatBotProdigy:
    def __init__(self, character_definition, characters_info_df):
        self.character_definition = character_definition
        self.characters_info_df = characters_info_df
    
        self.num_context_memories = NUM_CONTEXT_MEMORY
        self.chat_history_key = "chat_history"
        self.input_key = "input"
        # self.context_key = "context"

        self.chain = self.create_chain(character_definition)

    def create_chain(self, character_definition):
        conv_memory = ConversationBufferMemory(
            memory_key=self.chat_history_key, input_key=self.input_key
        )
        # context_memory = ElasticSearchRetrieverMemory(
        #     character_id=character_definition.character_id,
        #     characters_info_df=self.characters_info_df,
        #     memory_key=self.context_key,
        #     output_prefix=character_definition.name,
        #     blacklist=[self.chat_history_key],
        # )
        # print(context_memory)
        # # add the documents to the context memory
        # for i, summary in tqdm(enumerate(self.documents)):
        #     context_memory.save_context(inputs={}, outputs={f"[{i}]": summary})

        # print('-'*25 + 'CONTEXT MEMORY' + '-'*25)
        # print(context_memory)
        # print('-'*50)
        # context_memory_combined = "/n/n".join(context_memory)

        elastic_retriever = ElasticSearchRetriever(
            character_id=character_definition.character_id,
            characters_info_df=self.characters_info_df,
            k=NUM_CONTEXT_MEMORY
        )

        prompt = PromptTemplate.from_template(
            f"""Your name is {character_definition.name}.
Here is how you describe yourself:
---
{character_definition.long_description}
---

You will have a conversation with a Human, and you will engage in a dialogue with them.
You will exaggerate your personality, interests, desires, emotions, and other traits.
You will stay in character as {character_definition.name} throughout the conversation, even if the Human asks you questions that you don't know the answer to.
You will not break character as {character_definition.name}.
---
Current conversation:
---
{character_definition.name}: {character_definition.greeting}
{{{self.chat_history_key}}}
---

Human: {{{self.input_key}}}
{character_definition.name}:"""
        )

        GPT4o = ChatOpenAI(model='gpt-4o')
        conversation_chain = ConversationChain(
            llm=GPT4o, verbose=True, memory=conv_memory, prompt=prompt
        )
        chatbot = create_retrieval_chain(
            retriever=elastic_retriever,
            combine_docs_chain=conversation_chain,
        )
        return chatbot

    def greet(self):
        return self.character_definition.greeting

    def step(self, input):
        result = self.chain.invoke({'input': input, 'character_id': self.character_definition.character_id})
        print('-'*25 + 'CHAIN_RESULT' + '-'*25)
        pprint(result)
        print('-'*50)
        return result['answer']['response']
