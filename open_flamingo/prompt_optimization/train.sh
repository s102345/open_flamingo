python ./open_flamingo/prompt_optimization/manager.py \
--output_dir ./ \
--steps 5 \
--rices \
--cross_attn_every_n_layers 2 \
--shots 4 \
--example_rule "rand" \
--caption_number 3 \
--initial_prompt "Output: " \
--extra_information \

