import wget, gdown, os
import zipfile
import json


def download_files():
    root = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(root, 'data')

    if not os.path.exists(path):
        os.mkdir(path)

    # Download the COCO dataset

    # Annotations file
    if not os.path.exists(f'{path}/captions_train2014.json'):
        wget.download("https://huggingface.co/datasets/openflamingo/eval_benchmark/raw/main/mscoco_karpathy/annotations/captions_train2014.json", f'{path}/captions_train2014.json')
    
    # Karpathy splits with modification
    if not os.path.exists(f'{path}/prompt_karpathy_coco.json'):
        gdown.download('https://drive.google.com/u/3/uc?id=1WFzpbqHB7pH7KjPaJSKa0bk6Hzk5ztZF&export=download', f'{path}/prompt_karpathy_coco.json', quiet=False)

    # Download the COCO2014 dataset
    if not os.path.exists(f'{path}/train2014'):
        wget.download("http://images.cocodataset.org/zips/train2014.zip", f'{path}/train2014.zip')
        with zipfile.ZipFile(f'{path}/train2014.zip',"r") as zip_ref:
            zip_ref.extractall(f"{path}")
        os.remove(f'{path}/train2014.zip')

def make_split():
    if not os.path.exists(f'prompt_optimization/data/prompt_train2014'):
        os.mkdir(f'prompt_optimization/data/prompt_train2014')
        prompt_karpathy_coco = json.load(open('prompt_optimization/data/prompt_karpathy_coco.json'))
        prompt_train_fileName = []

        for img in prompt_karpathy_coco['images']:
            if img['split'] == 'test':
                prompt_train_fileName.append(img['filename'])

        # Transfer to new folder
        for fileName in prompt_train_fileName:
            os.rename(f'prompt_optimization/data/train2014/{fileName}', f'prompt_optimization/data/prompt_train2014/{fileName}')

def make_dataset():
    print("Downloading files...")
    download_files()
    print("Making split...")
    make_split()
    print("Done!")

if __name__ == '__main__':
    make_dataset()