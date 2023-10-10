import subprocess
import json

import os
os.environ["MASTER_ADDR"] = 'localhost'
os.environ["MASTER_PORT"] = '8888'

from make_dataset import make_dataset

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

def main():
    make_dataset()
    get_score("A photo of")

if __name__ == '__main__':
    main()