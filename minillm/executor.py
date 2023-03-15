import torch
import torch.nn as nn

from minillm.config import DEV, LLAMA_MODELS, OPT_MODELS, get_llm_config
from minillm.llms.llama.model import load_llama
from minillm.llms.opt.model import load_opt

def load_llm(model, weights):
    llm_config = get_llm_config(model)
    if model in LLAMA_MODELS:
        llm = load_llama(llm_config, weights)
    elif model in OPT_MODELS:
        llm = load_opt(llm_config, weights)
    else:
        raise ValueError(f"Invalid model name: {model}")
    llm.eval()
    return llm, llm_config

def generate(
    llm, llm_config, prompt, min_length, max_length, temperature, top_k, top_p
):
    from transformers import AutoTokenizer

    llm.to(DEV)
    tokenizer = AutoTokenizer.from_pretrained(llm_config.hf_config_name)
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(DEV)

    with torch.no_grad():
        generated_ids = llm.generate(
            input_ids,
            do_sample=True,
            min_length=min_length,
            max_length=max_length,
            top_p=top_p,
            top_k=top_k,
            temperature=temperature,
        )
    return tokenizer.decode([el.item() for el in generated_ids[0]])