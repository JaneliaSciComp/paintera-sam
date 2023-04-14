import sys

import cv2
import matplotlib.pyplot as plt
import numpy as np
from segment_anything import SamPredictor, sam_model_registry
import time


def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels == 1]
    neg_points = coords[labels == 0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white',
               linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white',
               linewidth=1.25)


def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30 / 255, 144 / 255, 255 / 255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def new_predictor(model_type="vit_h",
                  checkpoint="/home/hulbertc@hhmi.org/git/saalfeld/paintera_sam/sam_vit_h_4b8939.pth",
                  device="cuda") -> SamPredictor:
    model_type = model_type
    checkpoint = checkpoint
    sam = sam_model_registry[model_type](checkpoint=checkpoint)
    sam.to(device=device)
    return SamPredictor(sam)


def predict_current_image(predictor: SamPredictor, x: int, y: int, image, show: bool = False):
    input_points = np.array([[x, y]])
    input_labels = np.array([1])

    masks, scores, logits = predictor.predict(input_points, input_labels)
    
    if show:
        plt.figure(figsize=(10, 10))
        if image is not None:
            plt.imshow(image)
        show_mask(masks[0], plt.gca())
        plt.title(f"Mask, Score: {scores[0]:.3f}", fontsize=18)
        plt.axis('off')
        plt.show()
    return masks


def predict_new_image(img, x, y, out="/tmp/mask.png", show=False, predictor=None):
    if predictor == None:
        predictor=new_predictor()
        # model_type = "vit_h"
        # checkpoint = "/home/hulbertc@hhmi.org/git/saalfeld/paintera_sam/sam_vit_h_4b8939.pth"
        # sam = sam_model_registry[model_type](checkpoint=checkpoint)
        # sam.to(device="cuda")
        # predictor = SamPredictor(sam)

    print("Read image...", end="")
    start = time.time()
    image = cv2.imread(img)
    print(time.time() - start)

    print("Set image...", end="")
    start = time.time()
    predictor.set_image(image, image_format='BGR')
    print(time.time() - start)

    masks = predict_current_image(predictor, x, y, image, show=show)

    print(f"Saving to {out}...", end="")
    start = time.time()
    cv2.imwrite(out, masks[0] * 255)
    print(time.time() - start)

    return image


def main():
    begin = time.time()
    print("Initialize...", end="")
    start = time.time()
    model_type = "vit_h"
    checkpoint = "/home/hulbertc@hhmi.org/git/saalfeld/paintera_sam/sam_vit_h_4b8939.pth"
    sam = sam_model_registry[model_type](checkpoint=checkpoint)
    sam.to(device="cuda")
    print(time.time() - start)

    print("Load model...", end="")
    start = time.time()

    predictor = SamPredictor(sam)
    print(time.time() - start)
    print()

    start = time.time()
    image = predict_new_image("/tmp/sam.png", 225, 350, predictor=predictor, show=False)

    print(f"New Image Prediction: {time.time() - start}")
    print()

    start = time.time()
    image = predict_new_image("/tmp/sam.png", 225, 350, predictor=predictor, show=False)

    print(f"New Image Prediction: {time.time() - start}")
    print()

    start = time.time()
    image = predict_new_image("/tmp/sam.png", 225, 350, predictor=predictor, show=False)

    print(f"New Image Prediction: {time.time() - start}")
    print()

    print()
    print(f"Overall: {time.time() - begin}")

    print()
    print("Reused Image Predications:")

    for i in range(3):
        print(f"{i}\t", end="")
        predict_current_image(predictor, 225, 350, image, show=False)
    print()


if __name__ == "__main__":
    main()
