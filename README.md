# :parrot: MiniLLM: Large Language Models on Consumer GPUs

MiniLLM is a minimal system for running modern LLMs on consumer-grade GPUs. Its features include:

* Support for multiple LLMs (currently LLAMA, BLOOM, OPT) at various model sizes (up to 170B)
* Support for a wide range of consumer-grade NVIDIA GPUs
* Tiny and easy-to-use codebase mostly in Python (<500 LOC)

Underneath the hood, MiniLLM uses the the GPTQ algorithm for up to 3-bit compression and large reductions in GPU memory usage. See the hardware requirements for more information on which LLMs are supported by various GPUs.

### Goals

MiniLLM is a research project at Cornell Tech and Cornell University. Its goals are to:
* Provide an easy-to-use platform for creative experimentation with large language models
* Faciliate research on LLM alignment, bias mitigation, efficient inference, and other topics

## Installation

### Requirements

Any UNIX environment supporting Python (3.8 or greater) and PyTorch (we tested with 1.13.1+cu116) can run MiniLLM. See `requirements.txt` to reproduce our environment.

To ensure maximum reproducibility, consider creating an empty conda environment:
```
conda create -n minillm
conda activate minillm
conda install git pip virtualenv
```

### Setup

We use `distutils` to package MiniLLM. If you are not running conda, you can also create a `virtualenv`.
```
virtualenv minillm_env
source /minillm_env/bin/activate
pip install -r requirements.txt   # installs torch and two other packages
python setup.py install           # installs minillm in your environment
export CUDA_VISIBLE_DEVICES=0     # your GPU should be visible
```

Note that this process compiles and installs a custom CUDA kernel that is necessary to run quantized models.
We also use an experimental fork of the `transformers` library with support for LLAMA models.

## Running MiniLLM

The above process installs a `minillm` command in your environment.

### Download Models

First, start by downloading the weights of an LLM model:
```
minillm download --model llama-7b-4bit --weights llama-7b-4bit.pt
```
MiniLLM currently supports downloading `llama-7b-4bit` and `llama-13b-4bit`; we will add all the other LLAMA models this week.

### Generate Text

You can generate text directly from the command line:
```
minillm generate --model llama-7b-4bit --weights llama-7b-4bit.pt --prompt "the pyramids were built by"
```

The MiniLLM interface also provides additional command-line options.
```
usage: minillm generate [-h] --model {llama-7b-4bit,llama-13b-4bit} --weights WEIGHTS [--prompt PROMPT]
                        [--min-length MIN_LENGTH] [--max-length MAX_LENGTH] [--top_p TOP_P]
                        [--temperature TEMPERATURE]

options:
  -h, --help            show this help message and exit
  --model {llama-7b-4bit,llama-13b-4bit}
                        Type of model to load
  --weights WEIGHTS     Path to the model weights.
  --prompt PROMPT       Text used to initialize generation
  --min-length MIN_LENGTH
                        Minimum length of the sequence to be generated.
  --max-length MAX_LENGTH
                        Maximum length of the sequence to be generated.
  --top_p TOP_P         Top p sampling parameter.
  --temperature TEMPERATURE
                        Sampling temperature.
```

### Programmatic Usage

MiniLLM can also be used as a Python library:
```
import minillm.executor as minillm

llm, llm_config = minillm.load_llm('llama-7b-4bit', '/path/to/llama-7b-4bit.pt')
output = minillm.generate(
    llm, 
    llm_config, 
    prompt="the pyramids were built by", 
    min_length=10, 
    max_length=50, 
    top_p=0.95, 
    temperature=0.8,
)
print(output)
```

## Hardware Requirements

The following hardware is needed to run different models in MiniLLM:

| Model | GPU Memory Requirements | Compatible GPUs |
| ----- | -------------------- | --------------- |
| llama-7b-4bit | 6GB | RTX 2060, 3050, 3060 |
| llama-13b-4bit | 10GB | RTX 2060, 3060, 3080 |
| llama-33b-4bit | 20GB |  RTX 3080, A5000, 3090, 4090, V100 |
| llama-65b-4bit | 40GB | A100, 2x3090, 2x4090, A40, A6000 |

## Todos

This is experimental work in progress. We are working on adding:
* Out-of-the-box support for a additional LLMs. All the LLAMA models will be up by the end of the week.
* Automated quantization scripts
* Cleaning up the codebase

## Acknowledgements

MiniLLM is based on the following projects:
* The GPTQ algorithm and codebase by the [IST-DASLAB](https://github.com/IST-DASLab/gptq) with modifications by [@qwopqwop200](https://github.com/qwopqwop200/)
* The Transformer library, and its extension to use LLAMA models by [zphang](https://github.com/zphang/transformers/tree/llama_push)
* The LLAMA, OPT, and BLOOM models by META FAIR and the BigScience consortium.

## Feedback

Please send feedback to [Volodymyr Kuleshov](https://www.cs.cornell.edu/~kuleshov/).
