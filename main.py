from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import onnxruntime_genai as og
import argparse
import time

class Item(BaseModel):
	query: str
	
model = og.Model('cuda-int4-rtn-block-32')
tokenizer = og.Tokenizer(model)
tokenizer_stream = tokenizer.create_stream()
params = og.GeneratorParams(model)
search_options = {'max_length':2048,'do_sample': True,'temperature':1.0} 
params.set_search_options(**search_options)
chat_template = '<|user|>\n{input} <|end|>\n<|assistant|>'

app = FastAPI()

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
	"""
	generated_tokens = []
	while not generator.is_done():
		generator.compute_logits()
		generator.generate_next_token()
		new_token = generator.get_next_tokens()[0]
		generated_tokens.append(new_token)
		
	output = tokenizer_stream.decode(generated_tokens)
	return output
	"""

@app.post("/items/")
async def create_item(item: Item):
	
	query_dict = item.model_dump()
	query = query_dict['query']
	
	output = model_process(query)
	
	return {"response": output}