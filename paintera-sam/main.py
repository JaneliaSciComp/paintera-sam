import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
from segment_anything import build_sam, SamPredictor, sam_model_registry
import sys

def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels == 1]
    neg_points = coords[labels == 0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)


def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30 / 255, 144 / 255, 255 / 255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def predict(img, x, y, out="/tmp/mask.png", show=False, predictor=None):

    if predictor == None:
        predictor = SamPredictor(build_sam(checkpoint="sam_vit_h_4b8939.pth"))

    start = time.time()
    image = cv2.imread(img)
    # print(f"Read Image {time.time() - start}")
    start = time.time()
    predictor.set_image(image, image_format='BGR')
    print(f"Set Image {time.time() - start}")

    input_points = np.array([[x, y]])
    input_labels = np.array([1])

    if show:
        plt.figure(figsize=(10, 10))
        plt.imshow(image)
        show_points(input_points, input_labels, plt.gca())
        plt.axis('on')
        plt.show()


    start = time.time()
    masks, scores, logits = predictor.predict(input_points, input_labels)
    print(f"Predict {time.time() - start}")

    if show:
        for i, (mask, score) in enumerate(zip(masks, scores)):
            plt.figure(figsize=(10, 10))
            plt.imshow(image)
            show_mask(mask, plt.gca())
            plt.title(f"Mask {i + 1}, Score: {score:.3f}", fontsize=18)
            plt.axis('off')
            plt.show()

    start = time.time()
    cv2.imwrite(out, masks[0] * 255)
    # print(f"Write Image {time.time() - start}")

    print("done!", flush=True)
    print("ready!", flush=True)


if __name__ == "__main__":
    import time
    begining = time.time()
    print(f"Predicitng {sys.argv}, ")

    sam_checkpoint = "/home/hulbertc@hhmi.org/git/saalfeld/paintera-sam/sam_vit_h_4b8939.pth"
    model_type = "vit_h"
    device = "cuda"
    # device = "cpu"
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)

    start = time.time()
    print(f"Loading Model... ", end="")
    predictor = SamPredictor(sam)
    print(f"{time.time() - start}")
    start = time.time()
    print(f"Segmentation... ", end="")
    predict("/tmp/sam.png", 250, 350, predictor=predictor, show=False)
    print(f"{time.time() - start}")

    start = time.time()
    print(f"Segmentation... ", end="")
    predict("/tmp/sam.png", 250, 350, predictor=predictor, show=False)
    print(f"{time.time() - start}")

    start = time.time()
    print(f"Segmentation... ", end="")
    predict("/tmp/sam.png", 250, 350, predictor=predictor, show=False)
    print(f"{time.time() - start}")
    print(f"Overall: {time.time() - begining}")


