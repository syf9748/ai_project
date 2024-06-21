from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import onnxruntime_genai as og
import asyncio
import time

# define query format
class Item(BaseModel):
	query: str

# model initialization	
model = og.Model('cuda-int4-rtn-block-32')
tokenizer = og.Tokenizer(model)
tokenizer_stream = tokenizer.create_stream()
params = og.GeneratorParams(model)
search_options = {'max_length':2048,'do_sample': True,'temperature':1.0} 
params.set_search_options(**search_options)
systemq = "You are a customer service manager. You should politely answer the question from the customer."
chat_template = '<|system|>\n{system}<|end|><|user|>\n{input}<|end|>\n<|assistant|>'

app = FastAPI()

# feed input into the model
def model_process(query):
	prompt = f'{chat_template.format(system=systemq,input=query)}'
	input_tokens = tokenizer.encode(prompt)
	params.input_ids = input_tokens
	generator = og.Generator(model, params)
	
	output = []
	while not generator.is_done():
		generator.compute_logits()
		generator.generate_next_token()
		new_token = generator.get_next_tokens()[0]
		#store a list of tokens
		output.append(tokenizer_stream.decode(new_token))

	return output

@app.post("/items/")
async def create_item(item: Item):
	
	query_dict = item.model_dump()
	query = query_dict['query']
	# needs to be returned in json format
	output = {"response":model_process(query)}

	return output
