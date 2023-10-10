export PYTHONFAULTHANDLER=1
export CUDA_LAUNCH_BLOCKING=0

echo go $COUNT_NODE
echo $HOSTNAMES

export PYTHONPATH="$PYTHONPATH:open_flamingo"
python ./eval/scorer.py \
    --vision_encoder_path ViT-L-14 \
    --vision_encoder_pretrained openai \
    --lm_path anas-awadalla/mpt-1b-redpajama-200b \
    --lm_tokenizer_path anas-awadalla/mpt-1b-redpajama-200b \
    --checkpoint_path ./checkpoint.pt \
    --eval_coco \
    --device cuda:0 \
    --cross_attn_every_n_layers 1 \
    --results_file result.json \
    --num_samples -1 --shots 0 --num_trials 1 \
    --coco_train_image_dir_path ./train2014 \
    --coco_val_image_dir_path ./prompt_train2014 \
    --coco_karpathy_json_path ./prompt_karpathy_coco.json \
    --coco_annotations_json_path ./captions_train2014.json \
    --precision fp32 \
    --batch_size 2