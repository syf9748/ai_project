import streamlit as st
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import onnxruntime_genai as og

model = og.Model('cuda-int4-rtn-block-32')
tokenizer = og.Tokenizer(model)
tokenizer_stream = tokenizer.create_stream()
params = og.GeneratorParams(model)
search_options = {'max_length':2048,'do_sample': True,'temperature':1.0} 
params.set_search_options(**search_options)
chat_template = '<|user|>\n{input} <|end|>\n<|assistant|>'

def model_process(query):
	prompt = f'{chat_template.format(input=query)}'
	input_tokens = tokenizer.encode(prompt)
	params.input_ids = input_tokens
	generator = og.Generator(model, params)
	
	output = ''
	while not generator.is_done():
		generator.compute_logits()
		generator.generate_next_token()
		new_token = generator.get_next_tokens()[0]
		output = output + tokenizer_stream.decode(new_token)
	return output

prompt = st.chat_input("Say something")
if prompt:
    with st.chat_message("user"):
        st.write(prompt)

    output = model_process(prompt)

    with st.chat_message("ai"):
        st.write(output)