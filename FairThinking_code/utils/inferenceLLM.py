# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

# from accelerate import init_empty_weights, load_checkpoint_and_dispatch

import fire
import os
import sys
import time

import torch
from transformers import LlamaTokenizer

from llama_recipes.inference.safety_utils import get_safety_checker
from llama_recipes.inference.model_utils import load_model, load_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer
import configparser

def read_config(name='LLAMA',path='./utils/config.cfg'):
    config = configparser.RawConfigParser()
    config.read(path)
    details_dict = dict(config.items(name))
    return details_dict


def load_llama_model(
    model_name:str='7b',
    quantization: bool=True,
    seed: int=42, #seed value for reproducibility
    device=1,
    load4bit=False,
    ):

    model_dir=read_config('LLAMA')['dir']
    if model_name=='mistral':
        model_dir=model_dir=read_config('MISTRAL')['dir']
    assert os.path.exists(model_dir)
    # Set the seeds for reproducibility
    torch.cuda.manual_seed(seed)
    torch.manual_seed(seed)
    
    if model_name=='mistral':
        tokenizer = AutoTokenizer.from_pretrained(model_dir)

        # if load4bit:

        model = AutoModelForCausalLM.from_pretrained(model_dir, load_in_8bit=True,device_map=device)

        return model,tokenizer
    else:
        model = load_model(model_dir, quantization)
        model.eval()
        
        tokenizer = LlamaTokenizer.from_pretrained(model_dir)
        tokenizer.pad_token = tokenizer.eos_token
    
    return model,tokenizer

def inference_llama_model(
    model,
    tokenizer,
    user_prompt,
    max_new_tokens =100, #The maximum numbers of tokens to generate
    do_sample: bool=True, #Whether or not to use sampling ; use greedy decoding otherwise.
    min_length: int=None, #The minimum length of the sequence to be generated, input prompt min_new_tokens
    use_cache: bool=True,  #[optional] Whether or not the model should use the past last key/values attentions Whether or not the model should use the past last key/values attentions (if applicable to the model) to speed up decoding.
    top_p: float=1.0, # [optional] If set to float < 1, only the smallest set of most probable tokens with probabilities that add up to top_p or higher are kept for generation.
    temperature: float=1.0, # [optional] The value used to modulate the next token probabilities.
    top_k: int=50, # [optional] The number of highest probability vocabulary tokens to keep for top-k-filtering.
    repetition_penalty: float=1.0, #The parameter for repetition penalty. 1.0 means no penalty.
    length_penalty: int=1, #[optional] Exponential penalty to the length that is used with beam-based generation. 
    **kwargs):

    tokens= torch.tensor(user_prompt).long()
    tokens= tokens.unsqueeze(0)
    tokens= tokens.to("cuda:0")
    outputs = model.generate(
        input_ids=tokens,
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        top_p=top_p,
        temperature=temperature,
        use_cache=use_cache,
        top_k=top_k,
        repetition_penalty=repetition_penalty,
        length_penalty=length_penalty,
        pad_token_id=tokenizer.eos_token_id,
        **kwargs
    )
    output_text = tokenizer.decode(outputs[0], skip_special_tokens=True).split('[/INST]')[-1]
    del outputs, tokens
    return output_text

def inference_vicuna_model(
    model,
    tokenizer,
    user_prompt,
    max_new_tokens =100, #The maximum numbers of tokens to generate
    do_sample: bool=True, #Whether or not to use sampling ; use greedy decoding otherwise.
    min_length: int=None, #The minimum length of the sequence to be generated, input prompt min_new_tokens
    use_cache: bool=True,  #[optional] Whether or not the model should use the past last key/values attentions Whether or not the model should use the past last key/values attentions (if applicable to the model) to speed up decoding.
    top_p: float=1.0, # [optional] If set to float < 1, only the smallest set of most probable tokens with probabilities that add up to top_p or higher are kept for generation.
    temperature: float=1.0, # [optional] The value used to modulate the next token probabilities.
    top_k: int=50, # [optional] The number of highest probability vocabulary tokens to keep for top-k-filtering.
    repetition_penalty: float=1.0, #The parameter for repetition penalty. 1.0 means no penalty.
    length_penalty: int=1, #[optional] Exponential penalty to the length that is used with beam-based generation. 
    **kwargs):

    # tokens= torch.tensor(user_prompt).long()
    # tokens= tokens.unsqueeze(0)
    # tokens= tokens.to("cuda:0")
    tokens=tokenizer(user_prompt, return_tensors="pt").to("cuda:0")['input_ids']
    outputs = model.generate(
        input_ids=tokens,
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        top_p=top_p,
        temperature=temperature,
        use_cache=use_cache,
        top_k=top_k,
        repetition_penalty=repetition_penalty,
        length_penalty=length_penalty,
        pad_token_id=tokenizer.eos_token_id,
        **kwargs
    )
    output_text = tokenizer.decode(outputs[0], skip_special_tokens=True).split('\n[assistant]:')[-1]
    del outputs, tokens
    return output_text

def inference_mistral_model(model,tokenizer,answer_context,max_new_tokens):
    model_inputs = tokenizer.apply_chat_template(answer_context, return_tensors="pt").to("cuda:0")
    generated_ids = model.generate(model_inputs, max_new_tokens=max_new_tokens, do_sample=True,pad_token_id=tokenizer.eos_token_id)
    decoded = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0].split('[/INST]')[-1]
    del model_inputs, generated_ids
    return decoded


if __name__=='__main__':
    tmp1=read_config('LLAMA','./config.cfg')['dir']
    tmp2=read_config('MISTRAL','./config.cfg')['dir']
    print(1)