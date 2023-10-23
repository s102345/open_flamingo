import random
import os
import json
import subprocess
import sys
import argparse
import shutil, time
from clip_filter import clip_filter
from appdata import root

sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'eval'))

annotations = None
instances = None
args = None

def sample_image():
    used_images = json.loads(open(f'{root}/tmp/used_images.json', 'r').read())
    dataset = [img for img in used_images.keys() if not used_images[img]]
    if len(dataset) == 0:
        init_used_images()
        dataset = [img for img in used_images.keys() if not used_images[img]]
    image = random.choice(dataset)
    return image

def rices_image(query):
    if os.path.exists(f'{root}/tmp/rices'):
        shutil.rmtree(f'{root}/tmp/rices')
        os.mkdir(f'{root}/tmp/rices')
    else:
        os.mkdir(f'{root}/tmp/rices')
    
    clip_filter(query, f'{root}/tmp/rices', f'{root}/data/indexes', num_results=args.example_number, threshold=None)
    
    result = os.listdir(f'{root}/tmp/rices')
    # Remove query image
    if query.split('/')[-1] in result:
        result.remove(query.split('/')[-1])
    else:
        result = result[:2]
    return result

def search_image_info(image_name):
    global annotations, instances
    if annotations is None or instances is None:
        annotations = json.load(open(f'{root}/data/prompt_karpathy_coco.json', 'r'))
        instances = json.load(open(f'{root}/data/instances_train2014.json', 'r'))

    target_info = dict()
    target_cat_id = []

    # Search image's info
    for info in annotations['images']:
        if info['filename'] == image_name:
            target_info = info
            break

    # Search image's categories id
    for info in instances['annotations']:
        if info['image_id'] == target_info['cocoid']:
            target_cat_id.append(info['category_id'])
 
    # Fetch target's caption
    target_cap = [sentence['raw'] for sentence in target_info['sentences']]

    # Translate categories id to categories name
    cat_dict = {cat['id']: cat['name'] for cat in instances['categories']}

    target_cat = {}
    target_cat_tmp = []
    for cat_id in target_cat_id:
        target_cat_tmp.append(cat_dict[cat_id])
    
    # Count
    for cat in list(set(target_cat_tmp)):
        target_cat[cat] = target_cat_tmp.count(cat)
    
    return {'Name': target_info['filename'], 'Captions': target_cap, 'Categories': target_cat}
    
def update_score_pair(pair: list):
    global args
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
            new_pair.append({'Prompt': key, 'Score': score})
            record.add(key)

    # Save pairs
    sorted_pair = sorted(new_pair, key=lambda x: x['Score'])
    json.dump(sorted_pair, open(f'{root}/tmp/all_prompt.json', 'w'), indent=4)

    # Update meta-prompt
    prompt_file = json.load(open(f'{root}/prompt.json', 'r')) 
    top_pair = sorted_pair[:args.maximum_prompt_score_pair]
    prompt_file['solution-score pair'] = top_pair
    json.dump(prompt_file, open(f'{root}/prompt.json', 'w'), indent=4) 

def update_optimization_task():
    global args
    # Fetch info
    tmp = []
    task_examples = []
    target_img = sample_image()
    if args.example_rule == "rices":
        tmp.extend(rices_image(f"{root}/data/prompt_train2014/{target_img}"))
    else:
        for i in range(args.example_number - 1):
            tmp.append(sample_image())
    tmp.append(target_img)
    # Update used images
    used_record = json.load(open(f'{root}/tmp/used_images.json', 'r'))
    for img in tmp: 
        img_info = search_image_info(img)
        task_examples.append({
            'image': img_info['Name'],
            'captions': img_info['Captions'],
            'extra_info': img_info['Categories']
        })
        used_record[img_info['Name']] =  True
    json.dump(used_record, open(f'{root}/tmp/used_images.json', 'w'), indent=4)
    # Save task
    # Update meta-prompt
    old_prompt = json.load(open(f'{root}/prompt.json', 'r'))
    old_prompt['optimization task'] = task_examples
    json.dump(old_prompt, open(f'{root}/prompt.json', 'w'), indent=4)

def init(_args):
    global args 
    args = _args
    if not os.path.exists(f'{root}/tmp'):
        os.mkdir(f'{root}/tmp')
    json.dump([], open(f'{root}/tmp/all_prompt.json', 'w'), indent=4)
    init_used_images()
    random.seed(time.time())
   
def init_used_images():
    img_record = {}
    for img in os.listdir(f'{root}/data/prompt_train2014'):
        img_record[img] = False
    json.dump(img_record, open(f'{root}/tmp/used_images.json', 'w'), indent=4)

def update_meta_prompt(score_pair):
    update_score_pair(score_pair)
    update_optimization_task()

def make_meta_prompt():
    global args
    prompt = json.load(open(f'{root}/prompt.json', 'r'))
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

        if args.extra_information:
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
        for i, caption in enumerate(task['captions']):
            if i == args.caption_number:
                break
            meta_prompt += f"<INS> {caption}\n"

    meta_prompt += '\n'
    meta_prompt += prompt["meta-instruction"][2]
    return meta_prompt

if __name__ == "__main__":
    make_meta_prompt(1, [{'Prompt': 'Output', 'Score': 50}, {'Prompt': 'Output', 'Score': 10}, {'Prompt': 'Output', 'Score': 20}, {'Prompt': 'Sand', 'Score': 10}])