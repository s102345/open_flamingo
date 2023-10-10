import subprocess
import json
import wget
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

def download_checkpoint():
    if not os.path.exists('ckpt'):
        os.mkdir('ckpt')
    if not os.path.exists('ckpt/checkpoint.pt'):
        from huggingface_hub import hf_hub_download
        hf_hub_download("openflamingo/OpenFlamingo-3B-vitl-mpt1b-langinstruct", "ckpt/checkpoint.pt")

def main():
    make_dataset()
    download_checkpoint()
    get_score("A photo of")

if __name__ == '__main__':
    main()