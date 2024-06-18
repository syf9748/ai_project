import onnxruntime_genai as og
import argparse
import time

def main():
    

    model = og.Model('cuda-int4-rtn-block-32')
    tokenizer = og.Tokenizer(model)
    tokenizer_stream = tokenizer.create_stream()
    params = og.GeneratorParams(model)
    search_options = {'max_length':2048} 
    params.set_search_options(**search_options)
    chat_template = '<|user|>\n{input} <|end|>\n<|assistant|>'

    # Keep asking for input prompts in a loop
    
    text = "What's the weather today?"
    prompt = f'{chat_template.format(input=text)}'

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

       

if __name__ == "__main__":
    
    output=main()
    print(output)