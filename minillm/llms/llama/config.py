# NOTE: this configuration is experimental and will be modified to properly make use of HF Transformers
class LLama7B4BitConfig:
    hf_config_name = "decapoda-research/llama-7b-hf"
    weights_url = "https://huggingface.co/decapoda-research/llama-7b-hf-int4/resolve/main/llama-7b-4bit.pt"
    bits = 4

class LLama13B4BitConfig:
    hf_config_name = "decapoda-research/llama-13b-hf"
    weights_url = "https://huggingface.co/decapoda-research/llama-13b-hf-int4/resolve/main/llama-13b-4bit.pt"
    bits = 4

class LLama30B4BitConfig:
    hf_config_name = "decapoda-research/llama-30b-hf"
    weights_url = None
    bits = 4

class LLama65B4BitConfig:
    hf_config_name = "decapoda-research/llama-65b-hf"
    weights_url = None
    bits = 4    