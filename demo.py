# selecting object with SAM

import sys
# sys.path.append("..")
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from segment_anything import sam_model_registry, SamPredictor
from display import get_image, show_points, show_mask

sam_checkpoint = "../sam_ViT.models/sam_vit_b_01ec64.pth"
model_type = "vit_b"

device = "cuda"

sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)

predictor = SamPredictor(sam)

images_path = './notebooks/images/'
image_name = 'truck.jpg'
image = get_image(images_path + image_name)


predictor.set_image(image)

input_point = np.array([[500, 375]])
input_label = np.array([1])

plt.figure(figsize=(10,10))
# plt.imshow(image)
show_points(input_point, input_label, plt.gca())
plt.axis('on')
# plt.show()
plt.savefig(images_path+"display.jpg")

masks, scores, logits = predictor.predict(
    point_coords=input_point,
    point_labels=input_label,
    multimask_output=True,
)

print(masks.shape)  # (number_of_masks) x H x W


for i, (mask, score) in enumerate(zip(masks, scores)):
    plt.figure(figsize=(10,10))
    # plt.imshow(image)
    show_mask(mask, plt.gca())
    show_points(input_point, input_label, plt.gca())
    plt.title(f"Mask {i+1}, Score: {score:.3f}", fontsize=18)
    plt.axis('off')
    # plt.show()  
    plt.savefig(images_path+"scores.jpg")

