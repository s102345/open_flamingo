import subprocess
import json
import argparse
from pathlib import Path

import os
os.environ["MASTER_ADDR"] = 'localhost'
os.environ["MASTER_PORT"] = '8888'

from scorer import evaluate
from make_dataset import make_dataset
from meta_prompt import main as meta_prompt_main
import optimizer

root = os.path.dirname(os.path.abspath(__file__))
path = os.path.join(root, 'data')

def get_args():
    parser = argparse.ArgumentParser(description='OpenFlamingo Prompt Optimization')
    parser.add_argument('--output_dir', type=str, default="./", help='Output directory')
    # Training parameters
    parser.add_argument('--steps', type=int, default=200, help='Number of steps')
    parser.add_argument('--instruction_per_step', type=int, default=8, help='Instructions generated per step')
    parser.add_argument('--initial_prompt', type=str, default="Output", help='Initial prompt')
    # Meta-prompt parameters
    #parser.add_argument('--example_number', type=int, default=3, help='Example amount in meta prompt')
    parser.add_argument('--example_rule', type=str, default="rices", help='The way of choosing other 2 example in meta prompt')
    parser.add_argument('--caption_number', type=int, default=5, help='Caption amount of example in meta prompt')
    parser.add_argument('--maximum_score', type=int, default=-1, help='The maximum score given by scorer(Will be normalized)')
    parser.add_argument('--extra_information', action="store_true", help='Extra information of image in meta prompt')

def download_checkpoint():
    if not os.path.exists(f'{path}/checkpoint.pt'):
        print("Downloading checkpoint...")
        from huggingface_hub import hf_hub_download
        hf_hub_download("openflamingo/OpenFlamingo-3B-vitl-mpt1b-langinstruct", "checkpoint.pt", local_dir=path)

def update_path():
    #Update path to abs 
    scorer_params = json.load(open(f'{root}/scorer_params.json'))
    scorer_params['checkpoint_path'] = os.path.join(path, 'checkpoint.pt')
    scorer_params['coco_train_image_dir_path'] = f"{path}/train2014"
    scorer_params["coco_val_image_dir_path"] = f"{path}/prompt_train2014"
    scorer_params["coco_karpathy_json_path"] = f"{path}/prompt_karpathy_coco.json"
    scorer_params["coco_annotations_json_path"] = f"{path}/captions_train2014.json"
    json.dump(scorer_params, open(f'{root}/scorer_params.json', 'w'), indent=4)

def init():
    make_dataset()
    download_checkpoint()
    update_path()
    if not os.path.exists(f'{root}/tmp'):
        os.mkdir(f'{root}/tmp')

def generate_solution(instruction_per_step=8):
    #TODO: Call GPT-4
    solutions = []
    for i in range(instruction_per_step):
        sol = optimizer.generate()
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

def unit_test():
    print("Unit test")
    prompts = ["Output", "A Image of", "Output"]
    scores = evaluate(prompts)
    for score, prompt in zip(scores, prompts):
        print(f"Prompt: {prompt}, Score: {score}")
    print("Done")

def main():
    init()
    args = get_args()
    #meta_prompt_main(args)
    #unit_test()
    #train(args)

if __name__ == '__main__':
    main()