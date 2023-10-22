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
from meta_prompt import make_meta_prompt
from meta_prompt import init as meta_prompt_init
from optimizer import generate 
from optimizer import init as optimizer_init
from appdata import root, path

def get_args():
    parser = argparse.ArgumentParser(description='OpenFlamingo Prompt Optimization')

    parser.add_argument('--model_name_or_path', type=str, default="openflamingo/OpenFlamingo-3B-vitl-mpt1b-langinstruct", help='Model name or path')
    parser.add_argument('--output_dir', type=str, default="./", help='Output directory')

    # Training parameters
    parser.add_argument('--steps', type=int, default=200, help='Number of steps')
    parser.add_argument('--instruction_per_step', type=int, default=8, help='Instructions generated per step')
    parser.add_argument('--initial_prompt', type=str, default="Output", help='Initial prompt')

    # Meta-prompt parameters
    #parser.add_argument('--example_number', type=int, default=3, help='Example amount in meta prompt')
    parser.add_argument('--maximum_prompt_score_pair', type=int, default=20, help='Maximum number of prompt-score pair in meta prompt')
    parser.add_argument('--example_rule', type=str, default="rices", help='The way of choosing other 2 example in meta prompt')
    parser.add_argument('--caption_number', type=int, default=5, help='Caption amount of example in meta prompt')
    parser.add_argument('--maximum_score', type=int, default=-1, help='The maximum score given by scorer(Will be normalized)')
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

def rices_setup(indice_folder, images_path):
    data_dir = indice_folder + "/metadata/metadata_0.parquet"
    df = pd.read_parquet(data_dir)
    df['image_path'] = df['image_path'].apply(lambda row: row.replace("/content/prompt_train2014", images_path))
    df.to_parquet(data_dir)
 
def init(args):
    make_dataset()
    download_checkpoint(args.model_name_or_path)
    update_path()
    meta_prompt_init()
    optimizer_init()
    rices_setup(f'{path}/indexes', f'{path}/prompt_train2014') 

def generate_solution(meta_prompt, instruction_per_step=8):
    #TODO: Call GPT-4
    solutions = []
    optimizer_init()

    for i in range(instruction_per_step):
        sol = generate(meta_prompt)
        solutions.append(sol)

    return solutions

def train(args):
    for i in range(args.steps):
        #Generate solutions
        solutions = generate_solution(args.instruction_per_step)
        #Get scores
        scores = evaluate(solutions)
        #Update meta-prompts

        break

def main():
    args = get_args()
    init(args)
    meta_prompt = make_meta_prompt(args, [{'Prompt': 'Output', 'Score': 50}, {'Prompt': 'Output', 'Score': 10}, {'Prompt': 'Sand', 'Score': 10}])
    generate_solution(meta_prompt)

    #train(args)

if __name__ == '__main__':
    main()