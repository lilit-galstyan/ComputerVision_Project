import warnings
import os
import yaml
from diffusers import StableDiffusionPipeline
import torch

# setting fixed seed for reproducibility
seed = 123

cwd = os.getcwd()

yaml_file_path = cwd + "/Data/testing_data/test_texts.yml"

with open(yaml_file_path, "r") as file:
    test_cases = yaml.safe_load(file)["test_cases"]
# print(test_cases)


model_id = cwd + "/Code/model"
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float32, device_map=None)

# generating first image based on first test text
torch.manual_seed(seed + 21)
prompt1 = test_cases[0]
image = pipe(prompt1, num_inference_steps=50, guidance_scale=7.5).images[0]

image.save(cwd + "/Result/" + "rapunzel1.png")

# generating second image based on second test text
torch.manual_seed(seed + 999)
prompt2 = test_cases[1]
image = pipe(prompt2, num_inference_steps=50, guidance_scale=7.5).images[0]

image.save(cwd + "/Result/" + "rapunzel2.png")

# generating third image
torch.manual_seed(seed + 3333)
prompt3 = test_cases[2]
image = pipe(prompt3, num_inference_steps=50, guidance_scale=7.5).images[0]

image.save(cwd + "/Result/" + "rapunzel3.png")

# generating fourth image
torch.manual_seed(seed + 15)
prompt4 = test_cases[3]
image = pipe(prompt4, num_inference_steps=50, guidance_scale=7.5).images[0]

image.save(cwd + "/Result/" + "rapunzel4.png")

# generating fifth image
torch.manual_seed(seed + 757)
prompt5 = test_cases[4]
image = pipe(prompt5, num_inference_steps=70, guidance_scale=7.5).images[0]

image.save(cwd + "/Result/" + "rapunzel5.png")