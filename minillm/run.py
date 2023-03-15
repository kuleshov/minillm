import argparse
from minillm.config import LLM_MODELS

# ----------------------------------------------------------------------------

def make_parser():
    parser = argparse.ArgumentParser()
    parser.set_defaults(func=lambda args: parser.print_help())
    subparsers = parser.add_subparsers(title='Commands')

    # generate

    gen_parser = subparsers.add_parser('generate')
    gen_parser.set_defaults(func=generate)

    gen_parser.add_argument('--model', choices=LLM_MODELS, required=True,
        help='Type of model to load')
    gen_parser.add_argument('--weights', type=str, required=True,
        help='Path to the model weights.')
    gen_parser.add_argument('--prompt', type=str, default='',
        help='Text used to initialize generation')
    gen_parser.add_argument('--min-length', type=int, default=10, 
        help='Minimum length of the sequence to be generated.')
    gen_parser.add_argument('--max-length', type=int, default=50,
        help='Maximum length of the sequence to be generated.')
    gen_parser.add_argument('--top_p', type=float, default=.95,
        help='Top p sampling parameter.')
    gen_parser.add_argument('--top_k', type=int, default=50,
        help='Top p sampling parameter.')
    gen_parser.add_argument('--temperature', type=float, default=1.0,
        help='Sampling temperature.')

    # download

    dl_parser = subparsers.add_parser('download')
    dl_parser.set_defaults(func=download)

    dl_parser.add_argument('--model', choices=LLM_MODELS, required=True,
        help='Type of model to load')
    dl_parser.add_argument('--weights', type=str, default='./weights.pt',
        help='File where weights will be stored')

    return parser

# ----------------------------------------------------------------------------

def main():
    parser = make_parser()
    args = parser.parse_args()
    args.func(args)

def generate(args):
    import minillm.executor as minillm
    llm, llm_config = minillm.load_llm(args.model, args.weights)
    output = minillm.generate(
        llm, 
        llm_config, 
        args.prompt, 
        args.min_length, 
        args.max_length, 
        args.temperature,        
        args.top_k, 
        args.top_p, 
    )
    print(output)

def download(args):
    from minillm.config import get_llm_config
    from minillm.utils import download_file
    llm_config = get_llm_config(args.model)
    if not llm_config.weights_url:
        raise Exception(f"Downloading {args.model} is not supported")
    download_file(llm_config.weights_url, args.weights)

if __name__ == '__main__':
    main()    