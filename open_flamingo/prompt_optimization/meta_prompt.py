import os
import json
import sys
import random, time
from appdata import root
from sampler import Sampler
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'eval'))

class MetaPromptGenerator():
    def __init__(self, args, score_pair):
        self.args = args
        random.seed(time.time())
        self.sampler = Sampler()
        # Tmp of all prompt
        if not os.path.exists(f'{root}/tmp'):
            os.mkdir(f'{root}/tmp')
        json.dump([], open(f'{root}/tmp/all_prompt.json', 'w'), indent=4)
        # Init meta-prompt
        self.update_meta_prompt(score_pair)
    
    def update_score_pair(self, pair: list):
        # Read old prompt
        old_pair = json.load(open(f'{root}/tmp/all_prompt.json', 'r'))
        # Merge new pair
        old_pair.extend(pair)   
        pair_dict = {}
        pair_count = {}
        # Count showup times(for average)
        for p in old_pair:
            if p['Prompt'] not in pair_count:
                pair_count[p['Prompt']] = 1
            else:
                pair_count[p['Prompt']] += 1

        for p in old_pair:
            if p['Prompt'] not in pair_dict:
                pair_dict[p['Prompt']] = p['Score'] / pair_count[p['Prompt']]
            else:
                pair_dict[p['Prompt']] = pair_dict[p['Prompt']] + p['Score'] / pair_count[p['Prompt']]
        
        new_pair = []
        record = set()
        for key, score in pair_dict.items():
            if key not in record:
                new_pair.append({'Prompt': key, 'Score': round(score, self.args.round_off)})
                record.add(key)

        # Save pairs
        sorted_pair = sorted(new_pair, key=lambda x: x['Score'])
        json.dump(sorted_pair, open(f'{root}/tmp/all_prompt.json', 'w'), indent=4)

        # Update meta-prompt
        prompt_file = json.load(open(f'{root}/tmp/prompt.json', 'r')) 
        top_pair = sorted_pair[:self.args.maximum_prompt_score_pair]
        prompt_file['solution-score pair'] = top_pair
        json.dump(prompt_file, open(f'{root}/tmp/prompt.json', 'w'), indent=4) 

    def update_optimization_task(self):
        # Fetch info
        tmp = []
        task_examples = []
        target_img = self.sampler.sample_image()
        if self.args.example_rule == "rices":
            tmp.extend(self.sampler.rices_image(f"{root}/data/prompt_train2014/{target_img}", self.args.example_number))
        else:
            for i in range(self.args.example_number - 1):
                tmp.append(self.sampler.sample_image())
        tmp.append(target_img)
        # Update used images
        used_record = []
        for img in tmp: 
            img_info = self.sampler.search_image_info(img)
            task_examples.append({
                'image': img_info['Name'],
                'captions': img_info['Captions'],
                'extra_info': img_info['Categories']
            })
            used_record.append(img_info['Name'])
        self.sampler.update_record(used_record)
        # Save task
        # Update meta-prompt
        old_prompt = json.load(open(f'{root}/tmp/prompt.json', 'r'))
        old_prompt['optimization task'] = task_examples
        json.dump(old_prompt, open(f'{root}/tmp/prompt.json', 'w'), indent=4)

    def update_meta_prompt(self, score_pair):
        self.update_score_pair(score_pair)
        self.update_optimization_task()

    def generate_meta_prompt(self):
        prompt = json.load(open(f'{root}/tmp/prompt.json', 'r'))
        meta_prompt = ""

        # Comporse meta-prompt
        meta_prompt += prompt["meta-instruction"][0]
        meta_prompt += '\n\n'

        # Prompt-Score pair
        for pair in prompt['solution-score pair']:
            meta_prompt += f"prompt: {pair['Prompt']}, score: {pair['Score']}\n"
        meta_prompt += '\n'
        meta_prompt += prompt["meta-instruction"][1]
        meta_prompt += '\n\n'

        # Optimization task
        for i, task in enumerate(prompt['optimization task']):
            if i == len(prompt['optimization task']) - 1:
                meta_prompt += "Input: \n"
            else:
                meta_prompt += "In-context example:\n"

            meta_prompt += "Q:\n"

            if self.args.extra_information:
                meta_prompt += f"<IMG>, with extra info: "
                for info in task['extra_info']:
                    amount = task['extra_info'][info]
                    if info == list(task['extra_info'])[-1]:
                        meta_prompt += f"{amount} {info}"
                    else:
                        meta_prompt += f"{amount} {info}, "
                meta_prompt += '\n'
            else:
                meta_prompt += f"<IMG>\n"

            meta_prompt += 'A:\n'

            choosed_caption = random.choice(task['captions'], self.args.caption_number)
            for i, caption in enumerate(choosed_caption):
                meta_prompt += f"<INS> {caption}\n"

        meta_prompt += '\n'
        meta_prompt += prompt["meta-instruction"][2]
        return meta_prompt