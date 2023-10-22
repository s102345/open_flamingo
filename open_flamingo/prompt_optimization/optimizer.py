import openai
import os, json
from appdata import root

def init():
    json.dump([], open(f'{root}/tmp/solutions.json', 'w'))

def generate(meta_prompt):
    openai.api_key = os.getenv("OPENAI_API_KEY")

    messages = [
        {"role": "system", "content": '\n'.join(meta_prompt.split('\n')[:-1])},
        {"role": "user", "content": meta_prompt.split('\n')[-1]},
    ]

    past_solution = json.load(open(f'{root}/tmp/solutions.json', 'r'))
    for solution in past_solution:
        messages.append({"role": "assistant", "content": solution['solution']})
        
    completion  = openai.ChatCompletion.create(
        model='gpt-4',
        messages=messages
    )

    print(completion.choices[0].message)
    