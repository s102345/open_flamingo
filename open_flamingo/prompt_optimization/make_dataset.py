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
        print("Downloading annotation file...")
        wget.download("https://huggingface.co/datasets/openflamingo/eval_benchmark/raw/main/mscoco_karpathy/annotations/captions_train2014.json", f'{path}/captions_train2014.json', bar=wget.bar_adaptive)
    
    # Instances file
    if not os.path.exists(f'{path}/instances_train2014.json'):
        print("Downloading instance file...")
        gdown.download('https://drive.google.com/uc?id=1qgM2MUu2qhaacy64Ifm15WVLs1NG_5LE&export=download', f'{path}/instances_train2014.json', quiet=False)


    # Karpathy splits with modification
    if not os.path.exists(f'{path}/prompt_karpathy_coco.json'):
        print("Downloading splits file...")
        gdown.download('https://drive.google.com/u/3/uc?id=1WFzpbqHB7pH7KjPaJSKa0bk6Hzk5ztZF&export=download', f'{path}/prompt_karpathy_coco.json', quiet=False)

    # Download the COCO2014 dataset
    if not os.path.exists(f'{path}/train2014'):
        print("Downloading MSCOCO 2014 dataset...")
        wget.download("http://images.cocodataset.org/zips/train2014.zip", f'{path}/train2014.zip', bar=wget.bar_adaptive)
        with zipfile.ZipFile(f'{path}/train2014.zip',"r") as zip_ref:
            zip_ref.extractall(f"{path}")
        os.remove(f'{path}/train2014.zip')

def make_split():
    root = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(root, 'data')
    if not os.path.exists(f'{path}/prompt_train2014'):
        os.mkdir(f'{path}/prompt_train2014')
        prompt_karpathy_coco = json.load(open(f'{path}/prompt_karpathy_coco.json'))
        prompt_train_fileName = []

        for img in prompt_karpathy_coco['images']:
            if img['split'] == 'test':
                prompt_train_fileName.append(img['filename'])

        # Transfer to new folder
        for fileName in prompt_train_fileName:
            os.rename(f'{path}/train2014/{fileName}', f'{path}/prompt_train2014/{fileName}')

def make_dataset():
    print("Downloading files...")
    download_files()
    print("Making split...")
    make_split()
    print("Done!")

if __name__ == '__main__':
    make_dataset()