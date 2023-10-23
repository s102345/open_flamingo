import json
import argparse
from pathlib import Path
from huggingface_hub import hf_hub_download
import pandas as pd

import os
os.environ["MASTER_ADDR"] = 'localhost'
os.environ["MASTER_PORT"] = '8888'

from scorer import evaluate
from make_dataset import make_dataset
from meta_prompt import make_meta_prompt, update_meta_prompt
from meta_prompt import init as meta_prompt_init
from optimizer import generate
from optimizer import init as optimizer_init
from appdata import root, path

def get_args():
    parser = argparse.ArgumentParser(description='OpenFlamingo Prompt Optimization')

    # General parameters
    parser.add_argument('--output_dir', type=str, default="./", help='Output directory')
    parser.add_argument('--windows', action='store_true', help='Execute on Windows')

    # Model parameters
    parser.add_argument('--model_name_or_path', type=str, default="openflamingo/OpenFlamingo-3B-vitl-mpt1b-langinstruct", help='Model name or path')
    parser.add_argument('--rices', action='store_true', help='Use rices to evaluate score or not')
    parser.add_argument("--shots", nargs="+", default=[0, 4, 8, 16, 32], type=int)
    parser.add_argument("--num_trials", type=int, default=1, help="Number of trials to run for each shot using different demonstrations"    )
    parser.add_argument("--cross_attn_every_n_layers", type=int, default=1, help="Cross-attention every n layers")
    parser.add_argument("--lm_path", type=str, default="anas-awadalla/mpt-1b-redpajama-200b", help="Path to LLM")
    parser.add_argument("--lm_tokenizer_path", type=str, default="anas-awadalla/mpt-1b-redpajama-200b", help="Path to the tokenizer")

    # Training parameters
    parser.add_argument('--steps', type=int, default=200, help='Number of steps')
    parser.add_argument('--instruction_per_step', type=int, default=8, help='Instructions generated per step')
    parser.add_argument('--initial_prompt', type=str, default="Output", help='Initial prompt')

    # Meta-prompt parameters
    parser.add_argument('--example_number', type=int, default=3, help='Example amount in meta prompt')
    parser.add_argument('--maximum_prompt_score_pair', type=int, default=20, help='Maximum number of prompt-score pair in meta prompt')
    parser.add_argument('--example_rule', type=str, default="rices", help='The way of choosing other 2 example in meta prompt')
    parser.add_argument('--caption_number', type=int, default=5, help='Caption amount of example in meta prompt')
    #parser.add_argument('--maximum_score', type=int, default=-1, help='The maximum score given by scorer(Will be normalized)')
    parser.add_argument('--extra_information', action="store_true", help='Extra information of image in meta prompt')

    return parser.parse_args()

def download_checkpoint(model_name_or_path):
    if not os.path.exists(f'{path}/checkpoint.pt'):
        print("Downloading checkpoint...")
        hf_hub_download(model_name_or_path, "checkpoint.pt", local_dir=path)

def update_path():
    #Update path to abs 
    scorer_params = json.load(open(f'{root}/scorer_params.json'))
    scorer_params['checkpoint_path'] = os.path.join(path, 'checkpoint.pt')
    scorer_params['coco_train_image_dir_path'] = f"{path}/train2014"
    scorer_params["coco_val_image_dir_path"] = f"{path}/prompt_train2014"
    scorer_params["coco_karpathy_json_path"] = f"{path}/prompt_karpathy_coco.json"
    scorer_params["coco_annotations_json_path"] = f"{path}/captions_train2014.json"
    json.dump(scorer_params, open(f'{root}/scorer_params.json', 'w'), indent=4)

def update_scorer_args(args):
    params = json.load(open(f'{root}/scorer_params.json', 'r'))
    params['shots'] = args.shots
    params['num_trials'] = args.num_trials
    params['cross_attn_every_n_layers'] = args.cross_attn_every_n_layers
    params['rices'] = args.rices
    params['lm_tokenizer_path'] = args.lm_tokenizer_path
    params['lm_path'] = args.lm_path
    if args.windows:
        params['is_distributed'] = False
    else:
        params['is_distributed'] = True
    json.dump(params, open(f'{root}/scorer_params.json', 'w'), indent=4)

def rices_setup(indice_folder, images_path):
    data_dir = indice_folder + "/metadata/metadata_0.parquet"
    df = pd.read_parquet(data_dir)
    df['image_path'] = df['image_path'].apply(lambda row: row.replace("/content/prompt_train2014", images_path))
    df.to_parquet(data_dir)
 
def init(args):
    make_dataset()
    download_checkpoint(args.model_name_or_path)
    update_path()
    update_scorer_args(args)
    meta_prompt_init(args)
    optimizer_init()
    rices_setup(f'{path}/indexes', f'{path}/prompt_train2014')
    init_meta_prompt(args.initial_prompt) 

def generate_solution(meta_prompt, instruction_per_step=8):
    solutions = []
    optimizer_init()

    for i in range(instruction_per_step):
        sol = generate(meta_prompt)
        solutions.append(sol)

    return solutions

def train(args):
    for i in range(args.steps):
        # LOOP
        # Receive meta-prompt
        meta_prompt = make_meta_prompt()
        # Use meta-prompt to generate solutions
        solutions = generate_solution(meta_prompt)
        # Use solutions to get scores
        scores = [i * 10 for i in range(0, 8)]#evaluate(solutions)
        prompt_score_pair = []
        for sol, score in zip(solutions, scores):
            prompt_score_pair.append({'Prompt': sol, 'Score': score})
        #TODO: tmp
        update_meta_prompt(prompt_score_pair)
        break

def init_meta_prompt(init_prompt):
    score = 70.52 #evaluate(init_prompt)[0] 
    update_meta_prompt([{'Prompt': init_prompt, 'Score': score}])

def main():
    args = get_args()
    # Give a initial prompt & score
    # Update the meta-prompt
    init(args)
    train(args)

if __name__ == '__main__':
    main()