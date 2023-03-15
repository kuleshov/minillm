# NOTE: this configuration is experimental and will be modified to properly make use of HF Transformers
class OPT7B4BitConfig:
    hf_config_name = "facebook/opt-6.7b"
    weights_url = None
    bits = 4