import subprocess
import json
import argparse

import os
os.environ["MASTER_ADDR"] = 'localhost'
os.environ["MASTER_PORT"] = '8888'

from make_dataset import make_dataset
import optimizer

def get_args():
    parser = argparse.ArgumentParser(description='OpenFlamingo Prompt Optimization')
    # Training parameters
    parser.add_argument('--steps', type=int, default=200, help='Number of steps')
    parser.add_argument('--instruction_per_step', type=int, default=8, help='Instructions generated per step')
    parser.add_argument('--initial_prompt', type=str, default="Output", help='Initial prompt')
    # Meta-prompt parameters
    #parser.add_argument('--example_number', type=int, default=3, help='Example amount in meta prompt')
    parser.add_argument('--example_rule', type=str, default="rices", help='The way of choosing other 2 example in meta prompt')
    parser.add_argument('--maximum_score', type=int, default=-1, help='The maximum score given by scorer(Will be normalized)')
    parser.add_argument('--extra_information', action="store_true", help='Extra information of image in meta prompt')

def get_score(prompt: str):
    scorer_param = json.load(open('prompt_optimization\scorer_params.json'))
    parameters = ['python', './eval/scorer.py', '--prompt', prompt]
    for key, value in scorer_param.items():
        if value != "NONE":
            parameters.append(key)
            parameters.append(str(value))

    result = subprocess.run(parameters, capture_output=True)
    score = 0

    for line in result.stdout.splitlines():
        if line.startswith(b"Mean CIDEr score: "):
            score = float(line.split(b" ")[2:])
            break
    return score

def get_scores(prompts: list):
    scores = []
    for prompt in prompts:
        scores.append(get_score(prompt))
    return scores

def download_checkpoint():
    if not os.path.exists('ckpt'):
        os.mkdir('ckpt')
    if not os.path.exists('ckpt/checkpoint.pt'):
        from huggingface_hub import hf_hub_download
        hf_hub_download("openflamingo/OpenFlamingo-3B-vitl-mpt1b-langinstruct", "ckpt/checkpoint.pt")

def init():
    make_dataset()
    download_checkpoint()

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
    unit_test()
    #train(args)

if __name__ == '__main__':
    main()