# selecting object with SAM

import sys
# sys.path.append("..")
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import json

from segment_anything import sam_model_registry, SamPredictor, set_prompt
from display import get_image, show_points, show_mask

import torch

# sam_checkpoint = "../sam_ViT.models/sam_vit_b_01ec64.pth"
# model_type = "vit_b"

# sam_checkpoint = "../sam_ViT.models/sam_vit_l_0b3195.pth"
# model_type = "vit_l"

sam_checkpoint = "../sam_ViT.models/sam_vit_h_4b8939.pth"
model_type = "vit_h"

device = "cuda"

sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)
print("memory_allocated()", torch.cuda.memory_allocated(), "max_memory_allocated", torch.cuda.max_memory_allocated(), 
        "memory_reserved()", torch.cuda.memory_reserved())

predictor = SamPredictor(sam)

images_path = './notebooks/images/'
# image_name = 'groceries.jpg'
image_name = '023.jpeg'
# image_name = 'truck.jpg'
image = get_image(images_path + image_name)


predictor.set_image(image)
print("memory_allocated()", torch.cuda.memory_allocated(), "max_memory_allocated", torch.cuda.max_memory_allocated(), 
        "memory_reserved()", torch.cuda.memory_reserved())

# input_point = np.array([[500, 375]])
# input_label = np.array([1])
with open(".".join((images_path + image_name).split(".")[:-1]) + ".json") as f:
    prompt = json.load(f)
    input_point, input_label = set_prompt(prompt)

plt.figure(figsize=(10,10))
# plt.imshow(image)
show_points(input_point, input_label, plt.gca())
plt.axis('on')
# plt.show()
plt.savefig(images_path+"display.jpg")

i = 0
while i<1000:
    i += 1
    masks, scores, logits = predictor.predict(
        point_coords=input_point,
        point_labels=input_label,
        multimask_output=True,
    )
    if i % 100 == 0:
        # print(masks.shape)  # (number_of_masks) x H x W
        print("memory_allocated()", torch.cuda.memory_allocated(), "max_memory_allocated", torch.cuda.max_memory_allocated(), 
            "memory_reserved()", torch.cuda.memory_reserved())

for i, (mask, score) in enumerate(zip(masks, scores)):
    plt.figure(figsize=(10,10))
    # plt.imshow(image)
    show_mask(mask, plt.gca())
    show_points(input_point, input_label, plt.gca())
    plt.title(f"Mask {i+1}, Score: {score:.3f}", fontsize=18)
    plt.axis('off')
    # plt.show()  
    plt.savefig(images_path+"scores.jpg")
    