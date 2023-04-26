# selecting object with SAM

import sys
# sys.path.append("..")
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from segment_anything import sam_model_registry, SamPredictor
from display import get_image, show_points, show_mask

import torch

sam_checkpoint = "../sam_ViT.models/google/sam_ViT-L_16.pth"
model_type = "vit_l"

device = "cuda"

sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)
print("memory_allocated()", torch.cuda.memory_allocated(), "max_memory_allocated", torch.cuda.max_memory_allocated(), 
        "memory_reserved()", torch.cuda.memory_reserved())

images_path = './notebooks/images/'


def main(images_path):

    predictor = SamPredictor(sam)

    image_name = 'truck.jpg'
    image = get_image(images_path + image_name)

    # predictor.set_image(image)
    print("memory_allocated()", torch.cuda.memory_allocated(), "max_memory_allocated", torch.cuda.max_memory_allocated(), 
            "memory_reserved()", torch.cuda.memory_reserved())

    input_point = np.array([[500, 375]])
    input_label = np.array([1])

    plt.figure(figsize=(10,10))
    # plt.imshow(image)
    show_points(input_point, input_label, plt.gca())
    plt.axis('on')
    # plt.show()
    plt.savefig(images_path+"display.jpg")

    i = 0
    while i<10000:
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
    


main(images_path)