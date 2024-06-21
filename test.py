import streamlit as st
import requests
import json
import time

api_url = "http://127.0.0.1:8000/items/"

# sidebar
with st.sidebar:
    upload_file = st.file_uploader("Drop your file")
    st.header("File list:")
    if "file" not in st.session_state:
        st.session_state.file = []
    if upload_file is not None:
        st.session_state.file.append(upload_file.name)
    for file in st.session_state.file:
        st.write(file)
    
        
        

# title 
st.title("💬 Chatbot")
st.caption("🚀 A Streamlit chatbot powered by phi3 mini")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "How can I help you?"}]

# Display chat messages from history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# User input
if prompt := st.chat_input("Say something"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    # Display user input
    with st.chat_message("user"):
        st.markdown(prompt)

    # Display response
    with st.chat_message("assistant"):

        # Get response from FASTAPI
        response = requests.post(api_url, json = {"query":prompt})#, stream=True)

        if response.status_code == 200:
            #output = st.write_stream(response.iter_content())
            def generate_output(response):
                for item in response.json()["response"]:
                    yield item
                    time.sleep(0.1)
            output = st.write_stream(generate_output(response))
        else:
            output = st.write("Error: " + str(response.status_code))
    
    # Add response to chat history
    st.session_state.messages.append({"role": "assistant", "content": output})


