import random
import os
import json
import subprocess
import sys
import argparse
import shutil

sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'eval'))

root = os.path.dirname(os.path.abspath(__file__))

MAXIMUM_PAIR = 20

annotations = None
instances = None
args = None

def sample_image():
    dataset = os.listdir(f'{root}/data/prompt_train2014')
    image = random.choice(dataset)
    return search_image_info(image)

def rices_image(query):
    if os.path.exists(f'{root}/tmp/rices'):
        shutil.rmtree(f'{root}/tmp/rices')
        os.mkdir(f'{root}/tmp/rices')

    subprocess.run(["clip-retrieval", "filter", 
                    "--query", query, 
                    "--output_folder", f"{root}\\tmp\\rices",
                    "--indice_folder", f"{root}\\data\\indexes",
                    "--num_results", "2"])
    print(os.listdir(f'{root}/tmp/rices'))

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
    target_cat = []
    target_cat_tmp = []
    for cat_id in target_cat_id:
        for cat in instances['categories']:
            if cat['id'] == cat_id:
                target_cat_tmp.append(cat['name']) # Gathering categories name
    #Count
    for cat in list(set(target_cat_tmp)):
        tmp = {cat : target_cat_tmp.count(cat)}
        target_cat.append(tmp)

    return {'Name': target_info['filename'], 'Captions': target_cap, 'Categories': target_cat}

def update_score_pair(pair: list):
    # Read old prompt
    old_pair = json.load(open(f'{root}/tmp/all_prompt.json', 'r'))
    # Merge new pair
    old_pair.extend(pair)   
    keys = [p['Prompt'] for p in old_pair]
    scores = [p['Score'] for p in old_pair]
    
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            # Count average score 
            if keys[i] == keys[j]:
                scores[i] = (scores[i] + scores[j]) / 2
                scores[j] = scores[i]

    new_pair = []
    record = set()
    for key, score in zip(keys, scores):
        if key not in record:
            new_pair.append({'Prompt': key, 'Score': score})
            record.add(key)

    # Save pairs
    sorted_pair = sorted(new_pair, key=lambda x: x['Score'], reverse=True)
    json.dump(sorted_pair, open(f'{root}/tmp/all_prompt.json', 'w'), indent=4)

    # Update meta-prompt
    prompt_file = json.load(open(f'{root}/prompt.json', 'r')) 
    top_pair = sorted_pair[:MAXIMUM_PAIR]
    prompt_file['solution-score pair'] = top_pair
    json.dump(prompt_file, open(f'{root}/prompt.json', 'w'), indent=4) 

def update_optimization_task():
    target_img = sample_image()
    print(target_img)
    rices_image(f"{root}/data/prompt_train2014/{target_img['Name']}")

def init():
    if not os.path.exists(f'{root}/tmp'):
        os.mkdir(f'{root}/tmp')
    json.dump([], open(f'{root}/tmp/all_prompt.json', 'w'), indent=4)

def main(_args):
    global args
    args = _args
    init()
    update_score_pair([{'Prompt': 'Output', 'Score': 50}, {'Prompt': 'Output', 'Score': 10}, {'Prompt': 'Output', 'Score': 20}, {'Prompt': 'Sand', 'Score': 10}])
    update_score_pair([{'Prompt': 'Sand', 'Score': 40}, {'Prompt': 'Sand', 'Score': 10}])
    update_optimization_task()

if __name__ == "__main__":
    main(1)