import argparse
import importlib
import json
import os
import uuid
from collections import defaultdict

import sys
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'eval'))

import sys
import numpy as np
import torch
import utils


from coco_metric import compute_cider, postprocess_captioning_generation
from eval_datasets import CaptionDataset

from rices import RICES
from tqdm import tqdm

from eval_model import BaseEvalModel

from open_flamingo.src.flamingo import Flamingo

from open_flamingo.train.distributed import init_distributed_device, world_info_from_env
from models.open_flamingo import EvalModel

root = os.path.dirname(os.path.abspath(__file__))
path = os.path.join(root, 'data')

eval_model = None
cached_features = None
configs = dict

parser = argparse.ArgumentParser()

# Distributed evaluation
parser.add_argument(
    "--dist-url",
    default="env://",
    type=str,
    help="url used to set up distributed training",
)
parser.add_argument(
    "--dist-backend", default="nccl", type=str, help="distributed backend"
)
parser.add_argument(
    "--horovod",
    default=False,
    action="store_true",
    help="Use horovod for distributed training.",
)
parser.add_argument(
    "--no-set-device-rank",
    default=False,
    action="store_true",
    help="Don't set device index from local rank (when CUDA_VISIBLE_DEVICES restricted to one per proc).",
)

def init(args):
    global eval_model, cached_features
    global configs
    configs = json.load(open(f'{root}/scorer_params.json'))

    eval_model = EvalModel({
                            "vision_encoder_path": configs['vision_encoder_path'],
                            "vision_encoder_pretrained":  configs['vision_encoder_pretrained'],
                            "lm_path":  configs['lm_path'],
                            "lm_tokenizer_path":  configs['lm_tokenizer_path'],
                            "checkpoint_path":  configs['checkpoint_path'],
                            "cross_attn_every_n_layers":  configs['cross_attn_every_n_layers'],
                            "precision":  configs['precision'],
                            "device":  configs['device'],
                            })
    
    device_id = init_distributed_device(args)
    eval_model.set_device(device_id)
    eval_model.init_distributed()

    # load cached demonstration features for RICES
    if configs['cached_demonstration_features'] != 'NONE':
        cached_features = torch.load(
            f"{configs['cached_demonstration_features']}/coco.pkl", map_location="cpu"
        )
    else:
        cached_features = None

def evaluate_prompt(prompt="Output"):  
    # set up distributed evaluation
    args, _ = parser.parse_known_args()
    args.local_rank, args.rank, args.world_size = world_info_from_env()

    if eval_model is None:
        init(args)
    
    for shot in configs['shots']:
        scores = []
        for seed, trial in zip(configs['trial_seeds'], range(configs['num_trials'])):
            cider_score = evaluate_captioning(
                args=args,
                eval_model=eval_model,
                num_shots=shot,
                seed=seed,
                dataset_name="coco",
                cached_features=cached_features,
                test_prompt=prompt,
            )
        
            print(f"Shots {shot} Trial {trial} CIDEr score: {cider_score}")
            scores.append(cider_score)

        print(f"Shots {shot} Mean CIDEr score: {np.nanmean(scores)}")
        return np.nanmean(scores)


def evaluate_captioning(
    args: argparse.Namespace,
    eval_model: BaseEvalModel,
    seed: int = 42,
    min_generation_length: int = 0,
    max_generation_length: int = 20,
    num_beams: int = 3,
    length_penalty: float = 0.0,
    num_shots: int = 8,
    dataset_name: str = "coco",
    cached_features=None,
    test_prompt="Output",
):
    """Evaluate a model on COCO dataset.

    Args:
        args (argparse.Namespace): arguments
        eval_model (BaseEvalModel): model to evaluate
        seed (int, optional): seed for random number generator. Defaults to 42.
        max_generation_length (int, optional): maximum length of the generated caption. Defaults to 20.
        num_beams (int, optional): number of beams to use for beam search. Defaults to 3.
        length_penalty (float, optional): length penalty for beam search. Defaults to -2.0.
        num_shots (int, optional): number of in-context samples to use. Defaults to 8.
        dataset_name (str, optional): dataset to evaluate on. Can be "coco" or "flickr". Defaults to "coco".
        cached_features (tensor, optional): cached demonstration features for RICES. Defaults to None.
    Returns:
        float: CIDEr score

    """
    
    if dataset_name == "coco":
        image_train_dir_path = configs['coco_train_image_dir_path']
        image_val_dir_path = configs['coco_val_image_dir_path']
        annotations_path = configs['coco_karpathy_json_path']
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    train_dataset = CaptionDataset(
        image_train_dir_path=image_train_dir_path,
        image_val_dir_path=image_val_dir_path,
        annotations_path=annotations_path,
        is_train=True,
        dataset_name=dataset_name if dataset_name != "nocaps" else "coco",
    )

    test_dataset = CaptionDataset(
        image_train_dir_path=image_train_dir_path,
        image_val_dir_path=image_val_dir_path,
        annotations_path=annotations_path,
        is_train=False,
        dataset_name=dataset_name,
    )

    effective_num_shots = utils.compute_effective_num_shots(num_shots, 'open_flamingo')

    np.random.seed(seed)
    test_dataloader = utils.prepare_eval_samples(
        test_dataset,
        len(test_dataset),
        batch_size=configs['batch_size'],
    )

    if configs['rices']:
        rices_dataset = RICES(
            train_dataset,
            eval_model.device,
            batch_size=configs['batch_size'],
            cached_features=cached_features,
            vision_encoder_path=configs['vision_encoder_path'],
            vision_encoder_pretrained=configs['vision_encoder_pretrained'],
        )
    else:
        # subset of the training set to sample context images from
        query_set = utils.get_query_set(train_dataset, query_set_size=2048)

    predictions = defaultdict()
    for batch in tqdm(
        test_dataloader,
        desc=f"Running inference {dataset_name.upper()}",
    ):
        if configs['rices']:
            batch_demo_samples = rices_dataset.find(batch["image"], effective_num_shots)
        else:
            batch_demo_samples = utils.sample_batch_demos_from_query_set(
                query_set, effective_num_shots, len(batch["image"])
            )

        batch_images, batch_text = [], []
        for i in range(len(batch["image"])):
            if num_shots > 0:
                context_images = [x["image"] for x in batch_demo_samples[i]]
            else:
                context_images = []
            batch_images.append(context_images + [batch["image"][i]])

            context_text = "".join(
                [
                    eval_model.get_caption_prompt(prompt=test_prompt, caption=x["caption"].strip()) + "\n"
                    for x in batch_demo_samples[i]
                ]
            )

            # Keep the text but remove the image tags for the zero-shot case
            if num_shots == 0:
                context_text = context_text.replace("<image>", "")

            batch_text.append(context_text + eval_model.get_caption_prompt())

        outputs = eval_model.get_outputs(
            batch_images=batch_images,
            batch_text=batch_text,
            min_generation_length=min_generation_length,
            max_generation_length=max_generation_length,
            num_beams=num_beams,
            length_penalty=length_penalty,
        )

        new_predictions = [
            postprocess_captioning_generation(out).replace('"', "") for out in outputs
        ]

        for i, sample_id in enumerate(batch["image_id"]):
            predictions[sample_id] = {
                "caption": new_predictions[i],
            }

    # all gather
    all_predictions = [None for _ in range(args.world_size)]
    torch.distributed.all_gather_object(all_predictions, predictions)  # list of dicts

    if args.rank != 0:
        return None

    all_predictions = {
        k: v for d in all_predictions for k, v in d.items()
    }  # merge dicts

    # save the predictions to a temporary file
    results_path = f"{dataset_name}results_{uuid.uuid4()}.json"

    with open(results_path, "w") as f:
        f.write(
            json.dumps(
                [
                    {"image_id": k, "caption": all_predictions[k]["caption"]}
                    for k in all_predictions
                ],
                indent=4,
            )
        )

    metrics = compute_cider(
        result_path=results_path,
        annotations_path=configs['coco_annotations_json_path']
    )

    # delete the temporary file
    os.remove(results_path)

    return metrics["CIDEr"]

if __name__ == "__main__":
    init()
    print(evaluate_prompt('Output'))
