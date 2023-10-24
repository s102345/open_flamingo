import json
import argparse


import os
os.environ["MASTER_ADDR"] = 'localhost'
os.environ["MASTER_PORT"] = '8888'

from scorer import Scorer
import prompt_utils 
from meta_prompt import MetaPromptGenerator
from optimizer import Optimizer
from appdata import root, path

def get_args():
    parser = argparse.ArgumentParser(description='OpenFlamingo Prompt Optimization')

    # General parameters
    parser.add_argument('--output_dir', type=str, default="./", help='Output directory')
    parser.add_argument('--is_distributed', action='store_true', help='If execute on Windows platform, set this to False')

    # Model parameters
    parser.add_argument('--model_name_or_path', type=str, default="openflamingo/OpenFlamingo-3B-vitl-mpt1b-langinstruct", help='Model name or path')
    parser.add_argument('--rices', action='store_true', help='Use rices to evaluate score or not')
    parser.add_argument('--cached_demonstration_features', type=str, help='Cached demonstration features')
    parser.add_argument('--precision', type=str, default='fp32', help='Precision')
    parser.add_argument('--device', type=str, default='cuda:0', help='Device')
    parser.add_argument("--shots", nargs="+", default=[0, 4, 8, 16, 32], type=int)
    parser.add_argument("--num_trials", type=int, default=1, help="Number of trials to run for each shot using different demonstrations")
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
    parser.add_argument('--extra_information', action="store_true", help='Extra information of image in meta prompt')
    parser.add_argument('--round_off', type=int, default=2, help='Round off score in meta prompt')

    return parser.parse_args()

class Manager():
    def __init__(self, args):
        self.args = args
        
        print("Initializing...")
        prompt_utils.make_dataset()
        prompt_utils.download_checkpoint(self.args.model_name_or_path)
        prompt_utils.update_path()
        prompt_utils.update_scorer_args(self.args)
        prompt_utils.rices_setup()

        self.scorer = Scorer()
        self.optimizer = Optimizer()

        initial_score = 70.515#self.scorer.evaluate(args.initial_prompt)[0]
        self.metaPromptGenerator = MetaPromptGenerator(self.args, self.make_prompt_score_pair([self.args.initial_prompt], [initial_score])) 

    def make_prompt_score_pair(self, solutions, scores):
        prompt_score_pair = []
        for sol, score in zip(solutions, scores):
            prompt_score_pair.append({'Prompt': sol, 'Score': score})
        return prompt_score_pair
    
    def train(self):
        for i in range(self.args.steps):
            # LOOP
            # Receive meta-prompt
            meta_prompt = self.metaPromptGenerator.generate_meta_prompt()
            # Use meta-prompt to generate solutions
            solutions = []
            self.optimizer.init()
            for j in range(self.args.instruction_per_step):
                sol = self.optimizer.generate(meta_prompt)
                solutions.append(sol)
            # Use solutions to get scores
            scores = [i * 10 for i in range(0, 8)]#self.scorer.evaluate(solutions)
            prompt_score_pair = self.make_prompt_score_pair(solutions, scores)
            self.metaPromptGenerator.update_meta_prompt(prompt_score_pair)

def main():
    args = get_args()
    manager = Manager(args)
    manager.train()

if __name__ == '__main__':
    main()