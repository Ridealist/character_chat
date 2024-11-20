from typing import List
from pydantic import BaseModel, Field

from langchain_core.prompts import PromptTemplate
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.messages import BaseMessage, AIMessage
from langchain_core.chat_history import BaseChatMessageHistory

from langchain_openai import ChatOpenAI


class InMemoryHistory(BaseChatMessageHistory, BaseModel):
    """In memory implementation of chat message history."""

    messages: List[BaseMessage] = Field(default_factory=list)

    def add_messages(self, messages: List[BaseMessage]) -> None:
        """Add a list of messages to the store"""
        self.messages.extend(messages)

    def clear(self) -> None:
        self.messages = []


class BaseChatBotProdigy:
    def __init__(self, character_definition, characters_info_df):
        self.character_definition = character_definition
        self.characters_info_df = characters_info_df
    
        self.chat_history_key = "chat_history"
        self.input_key = "input"
        self.chat_history_store = {}

        self.chain = self.create_chain(character_definition)

    def set_chat_history(self, session_id: str, inmemory: InMemoryHistory) -> None:
        self.chat_history_store[session_id] = inmemory

    def get_chat_history(self, session_id: str) -> InMemoryHistory:
        if session_id not in self.chat_history_store:
            self.chat_history_store[session_id] = InMemoryHistory()
        return self.chat_history_store[session_id]

    def create_chain(self, character_definition):

        GPT4o = ChatOpenAI(model='gpt-4o')
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
If you don't know the answer, say that you don't know. Remember that this is a conversation with others, not a monologue to yourself, so keep it to a reasonable length.
Use TWO SENTECES MAXIMUM.
---
Human: {{{self.input_key}}}
{character_definition.name}:"""
        )
        conversational_chain = RunnableWithMessageHistory(
            runnable=qa_prompt | GPT4o, ##TODO OutputParser 설정하기!
            get_session_history=self.get_chat_history,
            input_messages_key="input",
            history_messages_key="chat_history",
            output_messages_key="answer",
            verbose=True
        )
        return conversational_chain

    def greet(self):
        return self.character_definition.greeting

    def step(self, input):
        result = self.chain.invoke(
            {
                "input": input,
                "character_id": self.character_definition.character_id,
            },
            config = {
                "configurable": {"session_id": "abc123"}
            }
        )
        # print(result)
        return result.content
