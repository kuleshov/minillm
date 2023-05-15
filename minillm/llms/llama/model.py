import torch

from minillm.utils import find_layers
from minillm.engine.converter import make_quant

def load_llama(llm_config, checkpoint):
    import transformers
    from transformers import LlamaConfig, LlamaForCausalLM
    def noop(*args, **kwargs):
        pass

    config = LlamaConfig.from_pretrained(llm_config.hf_config_name)
    torch.nn.init.kaiming_uniform_ = noop 
    torch.nn.init.uniform_ = noop 
    torch.nn.init.normal_ = noop 

    torch.set_default_dtype(torch.half)
    transformers.modeling_utils._init_weights = False
    torch.set_default_dtype(torch.half)
    model = LlamaForCausalLM(config)
    torch.set_default_dtype(torch.float)
    model = model.eval()
    layers = find_layers(model)
    for name in ['lm_head']:
        if name in layers:
            del layers[name]
    make_quant(model, layers, llm_config.bits)

    print('Loading LLAMA model')
    model.load_state_dict(torch.load(checkpoint))
    model.seqlen = 2048
    print('Done')

    return model
