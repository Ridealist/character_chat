from dataclasses import asdict
from io import StringIO
import json
import os
import streamlit as st
import pandas as pd

from data_driven_characters.character import generate_character_definition, Character, generate_character_definition_prodigy
from data_driven_characters.corpus import (
    generate_corpus_summaries,
    generate_docs,
)
from data_driven_characters.chatbots import (
    # SummaryChatBot,
    # RetrievalChatBot,
    # SummaryRetrievalChatBot,
    SummaryRetrievalChatBotProdigy
)
from data_driven_characters.interfaces import reset_chat, clear_user_input, converse

openai_api_key = st.secrets["openai_api_key"]
st.write(f'API Load Successfully : {st.secrets["openai_api_key"][-5:]}')
os.environ["OPENAI_API_KEY"] = openai_api_key

PRODIGY_DATAFRAME = None

@st.cache_resource()
def create_chatbot(character_definition, characters_info_df, chatbot_type): #corpus_summaries,)
    # if chatbot_type == "summary":
    #     chatbot = SummaryChatBot(character_definition=character_definition)
    # elif chatbot_type == "retrieval":
    #     chatbot = RetrievalChatBot(
    #         character_definition=character_definition,
    #         documents=corpus_summaries,
    #     )
    # elif chatbot_type == "summary with retrieval":
    #     chatbot = SummaryRetrievalChatBot(
    #         character_definition=character_definition,
    #         documents=corpus_summaries,
    #     )
    if chatbot_type == "summary with retrieval prodigy":
        chatbot = SummaryRetrievalChatBotProdigy(
            character_definition=character_definition,
            characters_info_df=characters_info_df
        )
        chatbot.character_definition = character_definition
        chatbot.characters_info_df = characters_info_df
    else:
        raise ValueError(f"Unknown chatbot type: {chatbot_type}")
    return chatbot


@st.cache_data(persist="disk")
def process_corpus(corpus):
    # load docs
    docs = generate_docs(
        corpus=corpus,
        chunk_size=2048,
        chunk_overlap=64,
    )

    # generate summaries
    corpus_summaries = generate_corpus_summaries(docs=docs, summary_type="map_reduce")
    return corpus_summaries


@st.cache_data(persist="disk")
def get_character_definition(name, corpus_summaries):
    character_definition = generate_character_definition(
        name=name,
        corpus_summaries=corpus_summaries,
    )
    return asdict(character_definition)


@st.cache_data(persist="disk")
def get_character_definition_prodigy(character_id, movie_id, movie_title, name, gender, mbti, biography):
    character_definition = generate_character_definition_prodigy(
        character_id=character_id,
        movie_id=movie_id,
        movie_title=movie_title,
        name=name,
        gender=gender,
        mbti=mbti,
        biography=biography
    )
    return asdict(character_definition)


@st.cache_data(persist="disk")
def load_prodigy():
    df = pd.read_json('data/characters.json', orient='index')
    PRODIGY_DATAFRAME = df.reset_index(names='character_id')
    return PRODIGY_DATAFRAME


def main():
    st.title("Persona-Consistent Character Chat")
    # st.write(
    #     "Upload a corpus in the sidebar to generate a character chatbot that is grounded in the corpus content."
    # )

    with st.sidebar:
        # uploaded_file = st.file_uploader("Upload corpus")
        # if uploaded_file is not None:
            # corpus_name = os.path.splitext(os.path.basename(uploaded_file.name))[0]

            # # read file
            # stringio = StringIO(uploaded_file.getvalue().decode("utf-8"))
            # corpus = stringio.read()

            # # scrollable text
            # st.markdown(
            #     f"""
            #     <div style='overflow: auto; height: 200px; border: 1px solid gray; border-radius: 5px; padding: 10px'>
            #         {corpus}</div>
            #     """,
            #     unsafe_allow_html=True,
            # )

            # st.divider()
            # get character name
            # character_name = st.text_input(f"Choose a character name from List")
            df = load_prodigy()

            movie_names = df['movie_name'].unique().tolist()
            movie_names_tp = tuple(movie_names)
            movie_select = st.selectbox(
                label="Choose a movie title from List",
                options=movie_names_tp,
                index=None
            )
            if movie_select:
                character_names = df[df["movie_name"] == movie_select]["character_name"].tolist()
                # character_names_firstCap = list(map(lambda char: char[0] + char[1:].lower(), character_names))
                character_name_tp = tuple(character_names)
                character_select = st.selectbox(
                    "Choose a character name from List",
                    character_name_tp,
                    index=None
                )

            if movie_select and character_select:
                if not openai_api_key:
                    st.error(
                        "You must enter an API key to use the OpenAI API. Please enter an API key in the sidebar."
                    )
                    return

                if (
                    "character_name" in st.session_state
                    and st.session_state["character_name"] != character_select
                ):
                    clear_user_input()
                    reset_chat()

                st.session_state["character_name"] = character_select

                # with st.spinner("Processing corpus (this will take a while)..."):
                #     corpus_summaries = process_corpus(corpus)

                with st.spinner("Generating character definition..."):
                    # get character definition
                    result = df.query(f"movie_name == '{movie_select}' and character_name == '{character_select}'")
                    character_definition = get_character_definition_prodigy(
                        character_id=result['character_id'].item(),
                        movie_id=result['movie_id'].item(),
                        movie_title=result['movie_name'].item(),
                        name=result['character_name'].item(),
                        gender=result['gender'].item(),
                        mbti=result['mbti'].item(),
                        biography=result['biography'].item(),
                    )
                    print(json.dumps(character_definition, indent=4))
                    # chatbot_type = st.selectbox(
                    #     "Select a memory type",
                    #     options=["summary with retrieval prodigy"],
                    #     # options=["summary", "retrieval", "summary with retrieval"],
                    #     index=0,
                    # )
                    # if (
                    #     "chatbot_type" in st.session_state
                    #     and st.session_state["chatbot_type"] != chatbot_type
                    # ):
                    #     clear_user_input()
                    #     reset_chat()

                    chatbot_type = "summary with retrieval prodigy"
                    st.session_state["chatbot_type"] = chatbot_type

                    # st.markdown(
                    #     f"[Export to character.ai](https://beta.character.ai/editing):"
                    # )
                    st.write(character_definition)

    if movie_select and character_select: #uploaded_file is not None and character_name:
        st.divider()
        chatbot = create_chatbot(
            character_definition=Character(**character_definition),
            characters_info_df=df,
            chatbot_type=chatbot_type,
            # corpus_summaries=corpus_summaries,
        )
        converse(chatbot)


if __name__ == "__main__":
    main()
