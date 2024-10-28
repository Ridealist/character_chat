from langchain.prompts import PromptTemplate
from langchain.chains.conversation.base import ConversationChain
from langchain.memory import ConversationBufferMemory

from langchain_openai import ChatOpenAI


class SummaryChatBot:
    def __init__(self, character_definition):
        self.character_definition = character_definition
        self.chain = self.create_chain(character_definition)

    def create_chain(self, character_definition):
        # GPT3 = ChatOpenAI(model_name="gpt-3.5-turbo")
        GPT4o = ChatOpenAI(model='gpt-4o')

        memory = ConversationBufferMemory(memory_key="chat_history", input_key="input")
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

Current conversation:
---
{character_definition.name}: {character_definition.greeting}
{{chat_history}}
---
Human: {{input}}
{character_definition.name}:"""
        )
        chatbot = ConversationChain(
            llm=GPT4o, verbose=True, memory=memory, prompt=prompt
        )
        return chatbot

    def greet(self):
        return self.character_definition.greeting

    def step(self, input):
        return self.chain.run(input=input)
