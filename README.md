# :parrot: MiniLLM: Large Language Models on Consumer GPUs

MiniLLM is a minimal system for running modern LLMs on consumer-grade GPUs. Its features include:

* Support for multiple LLMs (currently LLAMA, BLOOM, OPT) at various model sizes (up to 170B)
* Support for a wide range of consumer-grade Nvidia GPUs
* Tiny and easy-to-use codebase mostly in Python (<500 LOC)

Underneath the hood, MiniLLM uses the the GPTQ algorithm for up to 3-bit compression and large reductions in GPU memory usage. See the hardware requirements for more information on which LLMs are supported by various GPUs.

### Goals

MiniLLM is a research project at Cornell Tech and Cornell University. Its goals are to:
* Provide an easy-to-use platform for creative experimentation with large language models
* Faciliate research on LLM alignment, bias mitigation, efficient inference, and other topics

## Demo

This is MiniLLM running a quantized LLAMA-13B model on a NVIDIA GeForce GTX 1080 Ti:

```
$ minillm generate --model llama-13b-4bit --weights llama-13b-4bit.pt 
--prompt "In a shocking finding, scientist discovered a herd of unicorns living in a remote, previously 
unexplored valley, in the Andes Mountains. Even more surprising to the researchers was the fact that 
the unicorns spoke perfect English," --temperature 1. --top_k 50 --top_p 0.95 --max-length 500

In a shocking finding, scientist discovered a herd of unicorns living in a remote, previously unexplored valley, 
in the Andes Mountains. Even more surprising to the researchers was the fact that the unicorns spoke perfect English, 
and were quite willing to talk to the scientists. In their native tongue, scientists say that the unicorns, of which 
there are 19, are "fascinating creatures."

"We got a surprise when we first approached them, because they wanted to communicate with us," said Dr. Dirk-Erik 
Pruis, head of the international team of researchers, in a telephone interview. "Of course, we had to teach them to 
speak in English in the first place. At first they spoke a sort of Latin, but with a little coaching, they soon spoke 
proper English.

"Once we knew they could talk to us, we asked them about a great many things. It turns out that they're actually 
rather advanced animals. They speak poetry in Latin, for instance, and they all play the organ."
Asked about their origins, Pruis says they're "a little tough to explain" because scientists have determined that 
they are about 500 years old. They were first found by a prospector named Mr. J.M. Jones, who discovered the valley 
in 1939. He discovered the valley just before he passed away, and the animals have been isolated from the world ever 
since. The researchers are looking forward to learning much more about the unicorns and their valley.
```

This example is based on an old OpenAI [prompt](https://openai.com/research/better-language-models). See below for additional examples, including automatic essay generation and chain-of-thought prompting.

## Installation

### Requirements

Any UNIX environment supporting Python (3.8 or greater) and PyTorch (we tested with 1.13.1+cu116) can run MiniLLM. See `requirements.txt` for details.

To ensure maximum reproducibility, consider creating a new conda environment:
```
conda create -n minillm
conda activate minillm
conda install git pip virtualenv
# if you have not already installed CUDA in your system environment
# so that, for instance, nvcc is not in your PATH, then also:
conda install -c "nvidia/label/cuda-11.6.2" cuda-toolkit
```
MiniLLM also requries an NVIDIA GPU (Pascal architecture or newer); other platforms are currently unsupported.

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
You can also download the weights directly using `wget`:
```
wget https://huggingface.co/kuleshov/llama-30b-4bit/resolve/main/llama-30b-4bit.pt
wget https://huggingface.co/kuleshov/llama-65b-4bit/resolve/main/llama-65b-4bit.pt
```
The following models have pre-quantized weights: `llama-7b-4bit`, `llama-13b-4bit`, `llama-30b-4bit`, `llama-65b-4bit`.

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
| llama-13b-4bit | 10GB | GTX 1080, RTX 2060, 3060, 3080 |
| llama-30b-4bit | 20GB |  RTX 3080, A5000, 3090, 4090, V100 |
| llama-65b-4bit | 40GB | A100, 2x3090, 2x4090, A40, A6000 |

Only NVIDIA GPUs with the Pascal architecture or newer can run the current system.

## Additional Examples

In this example, the LLM produces an essay on the origins of the industrial revolution.
```
$ minillm generate --model llama-13b-4bit --weights llama-13b-4bit.pt --prompt "For today's homework assignment, please explain the causes of the industrial revolution." --temperature 1. --top_k 50 --top_p 0.95 --max-length 500
Loading LLAMA model
Done
For today's homework assignment, please explain the causes of the industrial revolution.
The Industrial Revolution was not a new invention. It was simply an expansion of an existing technology.
The major cause of the Industrial Revolution was the invention of the steam engine. Invented in England in the late 1700s by an engineer named James Watt, the steam engine was made possible by the development of the cylinder (which presses together a piston containing high pressure steam, creating an engine that uses steam as a power source). This new technology, which was not an invention of Thomas Newcomen in 1711, greatly influenced the Industrial Revolution.
Why did the Industrial Revolution occur in Great Britain rather than in other areas of Europe?
Although England is a small country (209,338 square kilometers or 80,758 square miles), it has a temperate climate that is not too hot or too cold, and its coastal location facilitates trade with other countries. Its proximity to the sea is ideal for fishing, and a well-developed road system, as well as the British canal system, provides rapid transportation within the country.
The country also has great reserves of coal and iron ore, which provide fuel for its factories. Iron is the most important product of the Industrial Revolution.
Moreover, the British enjoyed a political stability. England's long tradition of the rule of law and freedom of religion attracted many people from other countries.
The British government has a long tradition of being friendly to business. It gives foreign investment a friendly hand; indeed, Great Britain has the strongest private property protection of any country. It has a stable and long-standing economy; therefore, it can accept risk easily.
The British also have a long tradition of respect for innovation and invention. The English invented most of the basic technologies that were used in the 18th-century factories. The English value both science and engineering, and they were also willing to take risks and invest in new technology.
All of these factors, including a skilled and stable labor force, a reliable work ethic and a tradition of innovation, make Great Britain an ideal location for the first Industrial Revolution.
```

As expected, the LLMs support few-shot prompting. This is a demo of zero-shot translation.
```
$ minillm generate --model llama-13b-4bit --weights llama-13b-4bit.pt --prompt "English: This parrot is very loud. French:" --temperature 1. --top_k 50 --top_p 0.95 --max-length 500
Loading LLAMA model
Done
English: This parrot is very loud. French: Ce perroquet est fort bruyant. German: Dieser Papagei ist sehr laut. Italian: Questo Papagalli è molto rumoroso.
```

Interestingly, `llama-13b-4bit` is responsive to chain-of-thought prompting (though not perfectly):
```
$ minillm generate --model llama-13b-4bit --weights llama-13b-4bit.pt --prompt "Q: Roger has 5 tennis balls. He buys 2 more cans of tennis balls. Each can has 3 tennis balls. How many tennis balls does he have now? A: Roger started with 5 balls. 2 cans of 3 tennis balls each is 6 tennis balls. 5 + 6 = 11. The answer is 11. Q: A juggler can juggle 16 balls. Half of the balls are golf balls, and half of the golf balls are blue. How many blue golf balls are there?" --temperature 1. --top_k 50 --top_p 0.95 --max-length 400
Loading LLAMA model
Done
Q: Roger has 5 tennis balls. He buys 2 more cans of tennis balls. Each can has 3 tennis balls. How many tennis balls does he have now? 
A: Roger started with 5 balls. 2 cans of 3 tennis balls each is 6 tennis balls. 5 + 6 = 11. The answer is 11. 
Q: A juggler can juggle 16 balls. Half of the balls are golf balls, and half of the golf balls are blue. How many blue golf balls are there? 
A: We know that there are 16 balls. Half of the balls are golf balls. So we know that there are 8 golf balls. Half of the golf balls are blue, so that is 4 balls.
```
These examples were generated on an a NVIDIA GeForce GTX 1080 Ti using the `llama-13b-4bit` model. 

The 30B and 65B parameter models can also do zero-shot chain-of-thought reasoning (i.e., "let's think step-by-step"):
```
$ minillm generate --model llama-65b-4bit --weights /share/kuleshov/vk379/llama-65b-4bit.pt --prompt "Q: Roger has 5 tennis balls. He buys 2 more cans of tennis balls. Each can has 3 tennis balls. How many tennis balls does he have now? A: Let's think step-by-step."
Loading LLAMA model
Done
Q: Roger has 5 tennis balls. He buys 2 more cans of tennis balls. Each can has 3 tennis balls. 
How many tennis balls does he have now? A: Let's think step-by-step.
Roger has 5 balls
Roger bought 2 cans
Each can has 3 balls
So, Roger has 5 + 2 x 3 = 11 balls now!
```
Another example:
```
$ minillm generate --model llama-30b-4bit --weights /share/kuleshov/vk379/llama-30b-4bit.pt --prompt "Q: A juggler can juggle 16 balls. Half of the balls are golf balls, and half of the golf balls are blue. How many blue golf balls are there? A: Let's think step by step."
Loading LLAMA model
Done
Q: A juggler can juggle 16 balls. Half of the balls are golf balls, and half of the golf balls are blue. How many blue golf balls are there? A: Let's think step by step. There are 16 balls. The total number of golf balls is 16 / 2 = 8 The number of blue golf balls can be calculated by 8 * 1 / 2 = 4
```
In several cases, I generated 2-3 samples and took my favorite (i.e., not all samples were good).

## Todos

This is experimental work in progress. We are working on adding:
* Out-of-the-box support for a additional LLMs.
* Automated quantization scripts
* Cleaning up the codebase
* With a bit more time: fine-tuning models on consumer GPUs.

## Acknowledgements

MiniLLM is based on the following projects:
* The GPTQ algorithm and codebase by the [IST-DASLAB](https://github.com/IST-DASLab/gptq) with modifications by [@qwopqwop200](https://github.com/qwopqwop200/)
* The Transformer library, and its extension to use LLAMA models by [zphang](https://github.com/zphang/transformers/tree/llama_push)
* The LLAMA, OPT, and BLOOM models by META FAIR and the BigScience consortium.

## Citations

Please cite this repository if you use our system.

```
@misc{llmtune,
  author = {Volodymyr Kuleshov},
  title = {MiniLLM: Large Language Models on Consumer GPUs},
  year = {2023},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/kuleshov/minillm}},
}
```

## Feedback

Please send feedback to [Volodymyr Kuleshov](https://www.cs.cornell.edu/~kuleshov/).
