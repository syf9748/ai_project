from fastapi import FastAPI, UploadFile
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel
import onnxruntime_genai as og
import asyncio
import time
from sentence_transformers import SentenceTransformer
from pymilvus import MilvusClient, DataType
import uuid
from unstructured.partition.pdf import partition_pdf
from unstructured.chunking.title import chunk_by_title
from unstructured.staging.base import convert_to_dataframe
import json


# define query format
class Item(BaseModel):
	query: str

# model initialization	
model = og.Model('cuda-int4-rtn-block-32')
tokenizer = og.Tokenizer(model)
tokenizer_stream = tokenizer.create_stream()
params = og.GeneratorParams(model)
search_options = {'max_length':2048} 
params.set_search_options(**search_options)
systemq = "You are a customer service manager. You should use the information from context to answer customer's question."
#chat_template = '<|user|>\n{input}<|end|>\n<|assistant|>'
chat_template = '<|system|>\n{system}<|end|><|user|>\n{input}<|end|>\n<|assistant|>'

#Load transformer model
model_t = SentenceTransformer("all-MiniLM-L6-v2/")

def generate_uuid():
	id = int(uuid.uuid4())
	id = str(id)[:10]
	id = int(id)
	return id

app = FastAPI()

# feed input into the model
async def model_process(query):
	prompt = f'{chat_template.format(system=systemq,input=query)}'
	input_tokens = tokenizer.encode(prompt)
	params.input_ids = input_tokens
	generator = og.Generator(model, params)
	
	#output = []
	while not generator.is_done():
		generator.compute_logits()
		generator.generate_next_token()
		new_token = generator.get_next_tokens()[0]
		#store a list of tokens
		#output.append(tokenizer_stream.decode(new_token))
		yield tokenizer_stream.decode(new_token)

	#return output

@app.post("/chat/")
async def create_item(item: Item):
	
	query_dict = item.model_dump()
	query = query_dict['query']
	vectors = model_t.encode(query)

	client = MilvusClient("./milvus_demo.db")
	res = client.search(
		collection_name="demo_collection",
		data=[vectors],
		limit=3,
		output_fields=["text"],
	)
	result = ""
	for i in res[0]:
		result = result + " " + i["entity"]["text"]

	new_query = "question: " + query + "\n context: " + result
	client.close()
	# needs to be returned in json format
	#output = {"response":model_process(query)}

	#return output
	return StreamingResponse(model_process(new_query))

@app.post("/file/")
async def create_file(file: UploadFile):
	
	elements = partition_pdf(file=file.file,content_type="application/pdf", strategy ="fast")

	chunks = chunk_by_title(elements,new_after_n_chars=1000,max_characters=1000)
	df_chunk=convert_to_dataframe(chunks)
	lines = df_chunk['text'].to_list()
	
	#lines = []
	#for line in file.file:
		#lines.append(line.decode("utf-8"))

	vectors = model_t.encode(lines)

	data = [ {"id":generate_uuid(), "vector": vectors[i,:], "text": lines[i], "subject": "history"} for i in range(len(vectors)) ]
	
	#create milvus database
	client = MilvusClient("./milvus_demo.db")
	client.drop_collection(collection_name="demo_collection")
	client.create_collection(
		collection_name="demo_collection",
		dimension=384,  # The vectors we will use in this demo has 384 dimensions
	)


	res = client.insert(
		collection_name="demo_collection",
		data=data
	)

	# This will exclude any text in "history" subject despite close to the query vector.
	
	client.close()

	return({"answer":"Upload successful"})



