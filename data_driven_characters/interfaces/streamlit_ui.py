import streamlit as st


def reset_chat():
    st.cache_resource.clear()
    if "messages" in st.session_state:
        del st.session_state["messages"]


def clear_user_input():
    try:
        st.session_state["user_input"] = ""
    except:
        st.session_state["user_input"] = ""


def converse(chatbot):

    print('-'*50)
    print(st.session_state)
    print('-'*50)

    user_input = st.chat_input(
        # label=f"Chat with {chatbot.character_definition.name}",
        placeholder=f"Chat with {chatbot.character_definition.name}",
        # label_visibility="collapsed",
        key="user_input",
    )
    # left, right = st.columns([4, 1])

    if "messages" not in st.session_state:
        greeting = chatbot.greet()
        st.session_state["messages"] = [
            {
                "role": "assistant",
                "content": greeting,
                "key": 0,
            }
        ]
    # the old messages
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # the new message
    if user_input:
        with st.chat_message("user"):
            st.markdown(user_input)
        st.session_state.messages.append(
            {
                "role": "user",
                "content": user_input,
            }
        )

        with st.spinner(f"{chatbot.character_definition.name} is thinking..."):
            response = chatbot.step(user_input)

        with st.chat_message("assistant"):
            st.markdown(response)
        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": response,
            }
        )

    reset_chatbot = st.button("Reset") #, on_click=clear_user_input)
    if reset_chatbot:
        reset_chat()

class Streamlit:
    def __init__(self, chatbot):
        self.chatbot = chatbot

    def run(self):
        converse(self.chatbot)
