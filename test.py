from open_flamingo import create_model_and_transforms

model, image_processor, tokenizer = create_model_and_transforms(
    clip_vision_encoder_path="ViT-L-14",
    clip_vision_encoder_pretrained="openai",
    lang_encoder_path="anas-awadalla/mpt-1b-redpajama-200b",
    tokenizer_path="anas-awadalla/mpt-1b-redpajama-200b",
    cross_attn_every_n_layers=1,
)

from PIL import Image
import requests
import torch

model.load_state_dict(torch.load("open_flamingo\prompt_optimization\data\OpenFlamingo-3B-vitl-mpt1b-langinstruct.pt"), strict=False)

"""
Step 1: Load images
"""
demo_image_one = Image.open(
    requests.get(
        "http://farm4.staticflickr.com/3814/8961970060_1447d9fd08_z.jpg", stream=True
    ).raw
)

demo_image_two = Image.open(
    requests.get(
        "http://farm6.staticflickr.com/5510/9950700246_5591be9dcc_z.jpg",
        stream=True
    ).raw
)

query_image = Image.open(
    requests.get(
        "http://farm6.staticflickr.com/5510/9950700246_5591be9dcc_z.jpg", 
        stream=True
    ).raw
)


"""
Step 2: Preprocessing images
Details: For OpenFlamingo, we expect the image to be a torch tensor of shape 
 batch_size x num_media x num_frames x channels x height x width. 
 In this case batch_size = 1, num_media = 3, num_frames = 1,
 channels = 3, height = 224, width = 224.
"""
vision_x = [image_processor(demo_image_one).unsqueeze(0), image_processor(demo_image_two).unsqueeze(0), image_processor(query_image).unsqueeze(0)]
vision_x = torch.cat(vision_x, dim=0)
vision_x = vision_x.unsqueeze(1).unsqueeze(0)

"""
Step 3: Preprocessing text
Details: In the text we expect an <image> special token to indicate where an image is.
 We also expect an <|endofchunk|> special token to indicate the end of the text 
 portion associated with an image.
"""
tokenizer.padding_side = "left" # For generation padding tokens should be on the left
lang_x = tokenizer(
    ["""
        <image> Generate a descriptive caption for the provided image based on the given additional information and visual context: A gray, white and brown sheep has long hair blowing in the breeze.<|endofchunk|>
        <image> Generate a descriptive caption for the provided image based on the given additional information and visual context: A man takes a picture of snowy mountains with his cell phone.<|endofchunk|>
        <image> Generate a descriptive caption for the provided image based on the given additional information and visual context:
    """
    ],
    
     return_tensors="pt",
)


"""
Step 4: Generate text
"""
generated_text = model.generate(
    vision_x=vision_x,
    lang_x=lang_x["input_ids"],
    attention_mask=lang_x["attention_mask"],
    max_new_tokens=20,
    num_beams=3,
)

print("Generated text: ", tokenizer.decode(generated_text[0]))