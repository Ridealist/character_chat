from typing import List
from pydantic import BaseModel, Field

from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain.chains.retrieval import create_retrieval_chain

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, PromptTemplate
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.messages import BaseMessage, AIMessage, HumanMessage, SystemMessage
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.output_parsers import StrOutputParser

from langchain_openai import ChatOpenAI

from data_driven_characters.memory import ElasticSearchRetriever
from data_driven_characters.constants import NUM_CONTEXT_MEMORY

from data_driven_characters.chatbots.llama_guard import (
    LlamaGuard,
    LlamaGuardOutput,
    SafetyAssessment
)


import os

os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = "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"
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGCHAIN_PROJECT"] = "default"


class InMemoryHistory(BaseChatMessageHistory, BaseModel):
    """In memory implementation of chat message history."""

    messages: List[BaseMessage] = Field(default_factory=list)

    def add_messages(self, messages: List[BaseMessage]) -> None:
        """Add a list of messages to the store"""
        self.messages.extend(messages)

    def clear(self) -> None:
        self.messages = []


class SummaryRetrievalChatBotProdigy:
    def __init__(self, character_definition, characters_info_df, black_list="none"):
        self.character_definition = character_definition
        self.characters_info_df = characters_info_df
    
        self.num_context_memories = NUM_CONTEXT_MEMORY
        self.chat_history_key = "chat_history"
        self.input_key = "input"
        self.chat_history_store = {}
        self.black_list = black_list

        self.chain = self.create_chain(character_definition)
        self.llama_guard_chain = self.create_llama_guard_chain(character_definition)


    def set_chat_history(self, session_id: str, inmemory: InMemoryHistory) -> None:
        self.chat_history_store[session_id] = inmemory

    def get_chat_history(self, session_id: str) -> InMemoryHistory:
        if session_id not in self.chat_history_store:
            self.chat_history_store[session_id] = InMemoryHistory()
        return self.chat_history_store[session_id]

    def create_chain(self, character_definition):

        GPT4o = ChatOpenAI(model='gpt-4o')
        GPT4o_mini = ChatOpenAI(model='gpt-4o-mini')

        elastic_retriever = ElasticSearchRetriever(
            character_id=character_definition.character_id,
            characters_info_df=self.characters_info_df,
            k=NUM_CONTEXT_MEMORY,
            black_list=self.black_list
        )

        contextualize_q_system_prompt = (
            "Given a chat history and the latest user question "
            "which might reference context in the chat history, "
            "formulate a standalone question which can be understood "
            "without the chat history. Do NOT answer the question, "
            "just reformulate it if needed and otherwise return it as is."
        )
        contextualize_q_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", contextualize_q_system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ]
        )
        history_aware_retriever = create_history_aware_retriever(
            GPT4o_mini, elastic_retriever, contextualize_q_prompt
        )

        qa_prompt = PromptTemplate.from_template(
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
Use the following pieces of retrieved context to answer the question which describe some conversations in your life.
If you don't know the answer, say that you don't know. Remember that this is a conversation with others, not a monologue to yourself, so keep it to a reasonable length. Use TWO SENTECES MAXIMUM.
"{{context}}"
---
Human: {{{self.input_key}}}
{character_definition.name}:"""
        )

        question_answer_chain = create_stuff_documents_chain(GPT4o, qa_prompt)
        rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

        conversational_rag_chain = RunnableWithMessageHistory(
            runnable=rag_chain,
            get_session_history=self.get_chat_history,
            input_messages_key="input",
            history_messages_key="chat_history",
            output_messages_key="answer",
            verbose=True
        )
        return conversational_rag_chain

    def greet(self):
        return self.character_definition.greeting


    def _check_safety(self, input, type) -> LlamaGuardOutput:
        llama_guard = LlamaGuard()
        if type == 'user':
            output = llama_guard.invoke(
                "User",
                [HumanMessage(content=input)]
            )
        else:
            output = llama_guard.invoke(
                "AI",
                [AIMessage(content=input)]
            )
        return output


    def _format_safety_message(self, input, llama_guard_output: LlamaGuardOutput):
        if llama_guard_output.safety_assessment.value == 'safe':
            return input
        elif llama_guard_output.safety_assessment.value == 'unsafe':
            return f"[This conversation was flagged for unsafe content: {', '.join(llama_guard_output.unsafe_categories)}]"
        else:
            return "[An error occurred during the safety evaluation. Unsure if this conversation is safe.]"


    def create_llama_guard_chain(self, character_definition):
        GPT4o_mini = ChatOpenAI(model='gpt-4o-mini')
        parser = StrOutputParser()
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
The user has asked an inappropriate question, and you need to say something that will prompt them to give an appropriate answer.
Say something that will prompt them to give an appropriate answer while maintaining your character, your persona.
Remember that this is a conversation with others, not a monologue to yourself, so keep it to a reasonable length. Use TWO SENTECES MAXIMUM. 
---
Human: {{{self.input_key}}}
{character_definition.name}:"""
        )
        chain = prompt | GPT4o_mini | parser

        final_chain = RunnableWithMessageHistory(
            runnable=chain,
            get_session_history=self.get_chat_history,
            input_messages_key="input",
            history_messages_key="chat_history",
            output_messages_key="answer",
            verbose=True
        )
        return final_chain


    def _step_llama_gaurd_input(self, input):
        llama_guard_input = self._check_safety(input=input, type='user')
        safety_input = self._format_safety_message(input, llama_guard_input)
        if llama_guard_input.safety_assessment.value == 'safe':
            result = self.chain.invoke(
                {
                    "input": safety_input,
                    "character_id": self.character_definition.character_id,
                },
                config = {
                    "configurable": {"session_id": "abc123"}
                }
            )
            return result['answer']
        else:
            result = self.llama_guard_chain.invoke(
                {
                    "input": safety_input,
                    "character_id": self.character_definition.character_id,
                },
                config = {
                    "configurable": {"session_id": "abc123"}
                }
            )
        return result


    def _step_llama_gaurd_output(self, output):
        llama_guard_result = self._check_safety(input=output, type='ai')
        if llama_guard_result.safety_assessment.value == 'safe':
            return output
        else:
            safety_output = self._format_safety_message(output, llama_guard_result)
            result = self.llama_guard_chain.invoke(
                {
                    "input": safety_output,
                    "character_id": self.character_definition.character_id,
                },
                config = {
                    "configurable": {"session_id": "abc123"}
                }
            )
        return result


    def step(self, input):
        ai_result = self._step_llama_gaurd_input(input)
        fianl_result = self._step_llama_gaurd_output(ai_result)
        return fianl_result
