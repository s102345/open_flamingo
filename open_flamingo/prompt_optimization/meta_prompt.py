import random
import os

def sample_image():
    os.listdir('prompt_optimization/data/prompt_train2014')
    return random.choice(os.listdir('prompt_optimization/data/prompt_train2014'))