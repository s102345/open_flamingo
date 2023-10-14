import random
import os
import json

import sys
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'eval'))

root = os.path.dirname(os.path.abspath(__file__))

MAXIMUM_PAIR = 20

def sample_image():
    dataset = os.listdir(f'{root}/data/prompt_train2014')
    image = random.choice(dataset)
    print(search_image_info(image))


def search_image_info(image_name):
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
    target_cat = [instances['categories'][cat_id - 1]['name'] for cat_id in target_cat_id]
    target_cat = list(set(target_cat))

    return {'Captions': target_cap, 'Categories': target_cat}

def update_score_pair(pair: list):
    # Read old pair
    prompt_file = json.load(open(f'{root}/prompt.json', 'r')) 
    old_pair = prompt_file['solution-score pair']

    # Update old pair
    old_pair.extend(pair)
    sorted_pair = sorted(old_pair, key=lambda x: x['Score'], reverse=True)
    
    # Process
    duplicate_checker = set()
    for pair in sorted_pair:
        if pair['Prompt'] in duplicate_checker:
            sorted_pair.remove(pair)
        else:
            duplicate_checker.add(pair['Prompt'])

    new_pair = sorted_pair[:MAXIMUM_PAIR]
 
    # Write new pair
    prompt_file['solution-score pair'] = new_pair
    json.dump(prompt_file, open(f'{root}/prompt.json', 'w'), indent=4)

def update_optimization_task():
    print(sample_image())

def main():
    update_score_pair([{'Prompt': 'Output', 'Score': 50}])
    update_optimization_task()

if __name__ == "__main__":
    main()