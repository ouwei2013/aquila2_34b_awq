from threading import Thread

import torch
import gradio as gr
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, TextIteratorStreamer,AutoModelForCausalLM
from transformers.generation.utils import GenerationConfig
from peft import PeftModel
from awq import AutoAWQForCausalLM

from fastchat.model.model_adapter import get_conversation_template

# from sentence_transformers import SentenceTransformer,util
from transformers import AutoModel, AutoTokenizer
import gradio as gr
# import mdtex2html
import pandas as pd
import pickle
import numpy as np
import re
import tqdm
import torch
# from sentence_transformers import CrossEncoder,SentenceTransformer, util
# import numpy as np 

#from peft import PeftModel


model_id = "your/awq/file/path"
torch_device = "cuda" if torch.cuda.is_available() else "cpu"
print("Running on device:", torch_device)
print("CPU threads:", torch.get_num_threads())


if torch_device == "cuda":
    model = AutoAWQForCausalLM.from_quantized(model_id,trust_remote_code=True,fuse_layers=True)
    model.model.config.use_cache = True 
    tokenizer = AutoTokenizer.from_pretrained(model_id,trust_remote_code=True)
   

else:
    model = AutoModelForSeq2SeqLM.from_pretrained(model_id)
    tokenizer = AutoTokenizer.from_pretrained(model_id)


def get_prompt(user_text,history):
    conv = get_conversation_template("vicuna")
    conversations = []
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}
    sources =[]
    for query,response in history:
        sources.append({'from':'human','value':query})
        sources.append({'from':'gpt','value':response})
    sources.append({'from':'human','value':user_text})
    
    for j, sentence in enumerate(sources):
        role = roles[sentence["from"]]
        assert role == conv.roles[j % 2], f"{i}"
        conv.append_message(role, sentence["value"])
    return conv.get_prompt()+'ASSISTANT: '



def run_generation(text,chatbot,history,top_p, temperature, top_k, max_new_tokens):

    chatbot.append((text,""))
    user_text = get_prompt(text, history)
    print('-------------------------------')
    print(history)


    model_inputs = tokenizer([user_text], return_tensors="pt").to(torch_device)

    # Start generation on a separate thread, so that we don't block the UI. The text is pulled from the streamer
    # in the main thread. Adds timeout to the streamer to handle exceptions in the generation thread.
    streamer = TextIteratorStreamer(tokenizer, timeout=10., skip_prompt=True, skip_special_tokens=True)
    generate_kwargs = dict(
        model_inputs,
        streamer=streamer,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        top_p=top_p,
        temperature=float(temperature),
        top_k=top_k
    )

    t = Thread(target=model.generate, kwargs=generate_kwargs)
    t.start()

    # Pull the generated text from the streamer, and update the model output.
    model_output = ""
    for new_text in streamer:
        model_output += new_text
        # print(new_text)
        new_history = history + [(text, model_output)]
        chatbot[-1] = (text,model_output)
        yield chatbot, new_history
    # history.append((text,model_output))
    # return history,history


def reset_textbox():
    return gr.update(value='')


with gr.Blocks() as demo:
   
    chatbot = gr.Chatbot()
    state = gr.State([])
    
    with gr.Row():
        with gr.Column(scale=4):
            user_text = gr.Textbox(
                placeholder="Write an email about an alpaca that likes flan",
                label="User input"
            )
            # model_output = gr.Textbox(label="Model output", lines=10, interactive=False)
            button_submit = gr.Button(value="Submit")

        with gr.Column(scale=1):
            max_new_tokens = gr.Slider(
                minimum=1, maximum=1000, value=250, step=1, interactive=True, label="Max New Tokens",
            )
            top_p = gr.Slider(
                minimum=0.05, maximum=1.0, value=0.95, step=0.05, interactive=True, label="Top-p (nucleus sampling)",
            )
            top_k = gr.Slider(
                minimum=1, maximum=50, value=50, step=1, interactive=True, label="Top-k",
            )
            temperature = gr.Slider(
                minimum=0.1, maximum=5.0, value=0.8, step=0.1, interactive=True, label="Temperature",
            )

    user_text.submit(run_generation, [user_text,chatbot,state, top_p, temperature, top_k, max_new_tokens], [chatbot,state])
    button_submit.click(run_generation, [user_text,chatbot,state, top_p, temperature, top_k, max_new_tokens], [chatbot,state])

    demo.queue(max_size=32).launch(server_name='0.0.0.0',server_port=8501,enable_queue=True)