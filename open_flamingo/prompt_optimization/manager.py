import subprocess
import json
import argparse
from pathlib import Path

import os
os.environ["MASTER_ADDR"] = 'localhost'
os.environ["MASTER_PORT"] = '8888'

from scorer import evaluate_prompt
from make_dataset import make_dataset
import optimizer

root = os.path.dirname(os.path.abspath(__file__))
path = os.path.join(root, 'data')

def get_args():
    parser = argparse.ArgumentParser(description='OpenFlamingo Prompt Optimization')
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

def get_score(prompt: str):
    score = evaluate_prompt(prompt)
    return score

def get_scores(prompts: list):
    scores = []
    for prompt in prompts:
        scores.append(get_score(prompt))
    return scores

def download_checkpoint():
    print("Downloading checkpoint...")
    if not os.path.exists(f'{path}/checkpoint.pt'):
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
        scores = get_scores(solutions)
        #Update meta-prompts

        break

def unit_test():
    print("Unit test")
    prompts = ["Output", "Depicting the scene of", "A Image of", "Capturing a moment of", "Describe the scene where", "Output", "Output", "Output"]
    scores = get_scores(prompts)
    for score, prompt in zip(scores, prompts):
        print(f"Prompt: {prompt}, Score: {score}")
    print("Done")

def main():
    init()
    args = get_args()

    #unit_test()
    #train(args)

if __name__ == '__main__':
    main()