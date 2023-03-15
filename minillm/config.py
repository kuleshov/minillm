import torch
from minillm.llms.opt.config import OPT7B4BitConfig
from minillm.llms.llama.config import (
    LLama7B4BitConfig, LLama13B4BitConfig, LLama30B4BitConfig
)

# define some constants
DEV = torch.device('cuda:0')
LLAMA_MODELS = ["llama-7b-4bit", "llama-13b-4bit", "llama-30b-4bit"]
OPT_MODELS  = ["opt-6.7b-4bit"]
LLM_MODELS = LLAMA_MODELS + OPT_MODELS

# define some helpers
def get_llm_config(model):
    if model == "llama-7b-4bit":
        return LLama7B4BitConfig
    elif model == "llama-13b-4bit":
        return LLama13B4BitConfig
    elif model == "llama-30b-4bit":
        return LLama30B4BitConfig
    elif model == "opt-6.7b-4bit":
        return OPT7B4BitConfig      
    else:
        raise ValueError(f"Invalid model name: {model}")