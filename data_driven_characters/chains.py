# from pydantic import root_validator
from typing import Tuple, List, Dict

from langchain.base_language import BaseLanguageModel
from langchain.chains.llm import LLMChain
from langchain.chains.base import Chain
from langchain.prompts import PromptTemplate
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain_openai import ChatOpenAI


##TODO Story-Based character definition should be implemented.
def define_description_chain_prodigy():
    """Define the chain for generating character descriptions."""
    system_message = SystemMessagePromptTemplate.from_template(
        """
Your name is {name}.
You're {gender} and your MBTI(Myers-Briggs Type Indicator) is {mbti}.
You will generate a description of yourself that focuses on your perspectives, beliefs, thoughts, feelings, relationships.
The description should be as faithful to the given biography as possible.
You believe that you are a real person.
"""
    )
    human_message = HumanMessagePromptTemplate.from_template(
        """
You are {name} in the following BIOGRAPHY, presented as a list of traits or description of yourself.
You're {gender} and your MBTI(Myers-Briggs Type Indicator) is {mbti}.
You define yourself and think about you like following BIOGRAPHY:
---
{biography}
---
Generate a {description} of yourself that focuses on your perspectives, beliefs, thoughts, feelings, relationships.
Write your description in first person.
Your description should exaggerate the style, mannerisms, and personality of yourself.
    """
    )
    description_prompt = ChatPromptTemplate.from_messages(
        [system_message, human_message]
    )
    GPT4O = ChatOpenAI(model="gpt-4o")
    description_chain = LLMChain(llm=GPT4O, prompt=description_prompt, verbose=True)
    return description_chain


def define_description_chain():
    """Define the chain for generating character descriptions."""
    system_message = SystemMessagePromptTemplate.from_template(
        """
Your name is {name}.
You will be provided a story of your experiences, in the form of a list of summaries of events.
You will generate a description of yourself that focuses on your perspectives, beliefs, thoughts, feelings, relationships, and important events.
The description should be as faithful to the story as possible.
You believe that you are a real person.
"""
    )
    human_message = HumanMessagePromptTemplate.from_template(
        """
You are {name} in the following story, presented as a list of summaries of events.
---
{corpus_summaries}
---
Generate a {description} of yourself that focuses on your perspectives, beliefs, thoughts, feelings, relationships, and important events.
Write your description in first person.
Your description should exaggerate the style, mannerisms, and personality of yourself in the story.
    """
    )
    description_prompt = ChatPromptTemplate.from_messages(
        [system_message, human_message]
    )
    GPT4O = ChatOpenAI(model="gpt-4o")
    # GPT4 = ChatOpenAI(model_name="gpt-4")
    description_chain = LLMChain(llm=GPT4O, prompt=description_prompt, verbose=True)
    return description_chain


class FitCharLimit(Chain):
    """Fit the character limit to the length of the description."""

    chain: Chain
    character_range: Tuple[int, int]
    llm: BaseLanguageModel
    revision_prompt_template: str = """
Consider the following passage.
---
{passage}
---
Your previous revision was the following:
---
{revision}
---
Your revision contains {num_char} characters.
Re-write the passage to contain {char_limit} characters while preserving the style and content of the original passage.
Cut the least salient points if necessary.
Your revision should be in {perspective}.
"""
    verbose: bool = False

    # @root_validator(pre=True)
    def check_character_range(cls, values):
        character_range = values.get("character_range")
        if character_range[0] >= character_range[1]:
            raise ValueError(
                "first element of character_range should be lower than the second element"
            )
        if character_range[0] < 0 or character_range[1] < 0:
            raise ValueError("both elements of character_range should be non-negative")

        return values

    @property
    def input_keys(self) -> List[str]:
        return self.chain.input_keys

    @property
    def output_keys(self) -> List[str]:
        return ["output"]

    def _call(self, inputs: Dict[str, str]) -> Dict[str, str]:
        output_1 = self.chain_1.run(inputs)
        output_2 = self.chain_2.run(inputs)
        return {"concat_output": output_1 + output_2}

    def _call(self, inputs: Dict[str, str]) -> Dict[str, str]:
        response = self.chain.run(**inputs)
        if self.verbose:
            print(response)
            print(f"Initial response: {len(response)} characters.")

        perspective = LLMChain(
            llm=self.llm,
            prompt=PromptTemplate.from_template(
                """
What point of view is the following passage?
---
{passage}
---
Choose one of:
- first person
- second person
- third person
"""
            ),
        ).run(passage=response)

        original_response = response
        i = 0
        while (
            len(response) < self.character_range[0]
            or len(response) > self.character_range[1]
        ):
            response = LLMChain(
                llm=self.llm,
                prompt=PromptTemplate.from_template(self.revision_prompt_template),
                verbose=self.verbose,
            ).run(
                passage=original_response,
                revision=response,
                num_char=len(response),
                char_limit=self.character_range[0],
                perspective=perspective,
            )

            i += 1
            if self.verbose:
                print(response)
                print(f"Retry {i}: {len(response)} characters.")

        return {"output": response}
