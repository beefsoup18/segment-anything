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


sam_checkpoint = "../sam_ViT.models/google/sam_ViT-B_16.pth"
model_type = "vit_b"

# sam_checkpoint = "../sam_ViT.models/google/sam_ViT-L_16.pth"
# model_type = "vit_l"

device = "cuda"
num_epochs = 100

sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)
print("memory_allocated()", torch.cuda.memory_allocated(), "max_memory_allocated", torch.cuda.max_memory_allocated(), 
        "memory_reserved()", torch.cuda.memory_reserved())

images_path = '~/Downloads/SA-1B/sa_000000'


def main(images, prompts):

    trainer = SamTrainer(sam)

    for epoch in range(num_epochs):

        for i, (image, prompt) in enumerate(zip([images, prompts]))
            trainer.set_image(image)
            masks, scores, logits = trainer.train(
                point_coords=input_point,
                point_labels=input_label,
                multimask_output=True,
            )
            print(f"Epoch {i+1}/{len(images)}: ", end="\r")
        print()

        if epoch % 10 == 9:
            # print(masks.shape)  # (number_of_masks) x H x W
            print(f"Epoch {epoch+1}/{num_epochs}: ", "memory_allocated()", torch.cuda.memory_allocated(), 
                    "max_memory_allocated", torch.cuda.max_memory_allocated(), 
                    "memory_reserved()", torch.cuda.memory_reserved()
                    )

    trainer.save('trainer.pth')

    # for i, (mask, score) in enumerate(zip(masks, scores)):
    #     plt.figure(figsize=(10,10))
    #     # plt.imshow(image)
    #     show_mask(mask, plt.gca())
    #     show_points(input_point, input_label, plt.gca())
    #     plt.title(f"Mask {i+1}, Score: {score:.3f}", fontsize=18)
    #     plt.axis('off')
    #     # plt.show()  
    #     plt.savefig(images_path+"scores.jpg")


if __name__ == "__main__":

    images = []
    prompts = []

    for file_name in os.listdir(images_path):
        prompt_name = file_name[:-4] + ".json"
        if file_name[-4:] == ".jpg" and prompt_name in os.listdir(images_path):
            with open(images_path + "/" + prompt_name, "r") as f_prompt:
                images.append(get_image(images_path + "/" + file_name))
                prompts.append(np.array(f_prompt))

    if len(images) == len(prompts):
        main(images, prompts)
    else:
        print("Error in mismatch between images and prompts")