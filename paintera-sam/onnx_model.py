import time

import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
from onnxruntime import InferenceSession
from segment_anything import sam_model_registry, SamPredictor
from segment_anything.modeling import Sam
from segment_anything.utils.onnx import SamOnnxModel

import onnxruntime
from onnxruntime.quantization import QuantType
from onnxruntime.quantization.quantize import quantize_dynamic
import warnings


def show_mask(mask, ax):
    color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)

def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)

def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2))

h_checkpoint = "/home/hulbertc@hhmi.org/git/saalfeld/paintera-sam/sam_vit_h_4b8939.pth"
h_model_type = "vit_h"
h_onnx_model_path = "/home/hulbertc@hhmi.org/git/saalfeld/paintera-sam/sam_vit_h_4b8939.onnx"
h_quantized_onnx = "/home/hulbertc@hhmi.org/git/saalfeld/paintera-sam/sam_vit_h_4b8939_quantized.onnx"

b_checkpoint = "/home/hulbertc@hhmi.org/git/saalfeld/paintera-sam/sam_vit_b_01ec64.pth"
b_model_type = "vit_b"
b_onnx_model_path = "/home/hulbertc@hhmi.org/git/saalfeld/paintera-sam/sam_vit_b_01ec64.onnx"
b_quantized_onnx = "/home/hulbertc@hhmi.org/git/saalfeld/paintera-sam/sam_vit_b_01ec64_quantized.onnx"
def export_onnx_model(sam : Sam, onnx_model_path : str, quantized_onnx_model_path : str):
    sam_onnx_model = SamOnnxModel(sam, return_single_mask=True)

    dynamic_axes = {
        "point_coords": {1: "num_points"},
        "point_labels": {1: "num_points"},
    }

    embed_dim = sam.prompt_encoder.embed_dim
    embed_size = sam.prompt_encoder.image_embedding_size
    mask_input_size = [4 * x for x in embed_size]
    dummy_inputs = {
        "image_embeddings": torch.randn(1, embed_dim, *embed_size, dtype=torch.float),
        "point_coords": torch.randint(low=0, high=1024, size=(1, 5, 2), dtype=torch.float),
        "point_labels": torch.randint(low=0, high=4, size=(1, 5), dtype=torch.float),
        "mask_input": torch.randn(1, 1, *mask_input_size, dtype=torch.float),
        "has_mask_input": torch.tensor([1], dtype=torch.float),
        "orig_im_size": torch.tensor([1500, 2250], dtype=torch.float),
    }
    output_names = ["masks", "iou_predictions", "low_res_masks"]

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=torch.jit.TracerWarning)
        warnings.filterwarnings("ignore", category=UserWarning)
        with open(onnx_model_path, "wb") as f:
            torch.onnx.export(
                sam_onnx_model,
                tuple(dummy_inputs.values()),
                f,
                export_params=True,
                verbose=False,
                opset_version=17,
                do_constant_folding=True,
                input_names=list(dummy_inputs.keys()),
                output_names=output_names,
                dynamic_axes=dynamic_axes,
            )


    quantize_dynamic(
        model_input=onnx_model_path,
        model_output=quantized_onnx_model_path,
        optimize_model=True,
        per_channel=False,
        reduce_range=False,
        weight_type=QuantType.QUInt8,
    )

def load_model(sam : Sam, model_path) :
    ort_session = onnxruntime.InferenceSession(model_path)
    sam.to(device="cuda")
    # sam.to(device="cpu")
    return SamPredictor(sam), ort_session

def predict_current_image(predictor : SamPredictor, ort_session : InferenceSession, x : int, y : int, image, image_embedding, show=False):
    input_point = np.array([[x, y]])
    input_label = np.array([1])

    onnx_coord = np.concatenate([input_point, np.array([[0.0, 0.0]])], axis=0)[None, :, :]
    onnx_label = np.concatenate([input_label, np.array([-1])], axis=0)[None, :].astype(np.float32)

    onnx_coord = predictor.transform.apply_coords(onnx_coord, image.shape[:2]).astype(np.float32)
    onnx_mask_input = np.zeros((1, 1, 256, 256), dtype=np.float32)
    onnx_has_mask_input = np.zeros(1, dtype=np.float32)

    ort_inputs = {
        "image_embeddings": image_embedding,
        "point_coords": onnx_coord,
        "point_labels": onnx_label,
        "mask_input": onnx_mask_input,
        "has_mask_input": onnx_has_mask_input,
        "orig_im_size": np.array(image.shape[:2], dtype=np.float32)
    }

    print("Segment...", end="")
    start = time.time()
    masks, _, low_res_logits = ort_session.run(None, ort_inputs)
    masks = masks > predictor.model.mask_threshold
    print(time.time() - start)

    if show:
        plt.figure(figsize=(10,10))
        plt.imshow(image)
        show_mask(masks, plt.gca())
        show_points(input_point, input_label, plt.gca())
        plt.axis('off')
        plt.show()

    return masks
def predict_new_image(img : str, x : int, y : int, out="/tmp/mask.png", predictor : SamPredictor=None, ort_session : onnxruntime.InferenceSession=None, show=False):

    if predictor is None or ort_session is None:
        predictor, ort_session = load_model()


    print("Read image...", end="")
    start = time.time()
    image = cv2.imread(img)
    print(time.time() - start)

    print("Set image...", end="")
    start = time.time()
    predictor.set_image(image, image_format='BGR')
    print(time.time() - start)

    print("Get image embedding...", end="")
    start = time.time()
    image_embedding = predictor.get_image_embedding().cpu().numpy()
    print(time.time() - start)

    masks = predict_current_image(predictor, ort_session, x, y, image, image_embedding, show=show)

    print(f"Saving to {out}...", end="")
    start = time.time()
    h, w = masks[0].shape[-2:]
    mask_image = masks[0].reshape(h, w, 1) * 255
    cv2.imwrite(out, mask_image)
    print(time.time() - start)

    return image, image_embedding


if __name__ == '__main__':
    begin = time.time()
    print("Initialize...", end="")
    start = time.time()
    model_type = h_model_type
    checkpoint = h_checkpoint
    sam = sam_model_registry[model_type](checkpoint=checkpoint)
    print(time.time() - start)

    onnx = h_onnx_model_path
    quantized_onnx = h_quantized_onnx
    # export_onnx_model(sam, onnx, quantized_onnx)

    print("Load model...", end="")
    start = time.time()
    (predictor, ort_session) = load_model(sam, onnx) #quantized_onnx)
    print(time.time() - start)
    print()

    # for i in range(20):

    start = time.time()
    (image, image_embedding) = predict_new_image("/tmp/sam.png", 225, 350, predictor=predictor, ort_session=ort_session, show=False)

    print(f"New Image Prediction: {time.time() - start}")
    print()

    start = time.time()
    (image, image_embedding) = predict_new_image("/tmp/sam.png", 225, 350, predictor=predictor, ort_session=ort_session, show=False)

    print(f"New Image Prediction: {time.time() - start}")
    print()

    start = time.time()
    (image, image_embedding) = predict_new_image("/tmp/sam.png", 225, 350, predictor=predictor, ort_session=ort_session, show=False)

    print(f"New Image Prediction: {time.time() - start}")
    print()

    print()
    print(f"Overall: {time.time() - begin}")

    # print()
    # print("Reused Image Predications:")
    #
    # for i in range(3):
    #     print(f"{i}\t", end="")
    #     predict_current_image(predictor, ort_session, 225, 350, image, image_embedding, show=False)
    # print()

    # print("Show")
    # run_predict(predictor, ort_session, 425, 350, image, image_embedding, show=True)
    # run_predict(predictor, ort_session, 225, 550, image, image_embedding, show=True)
    # run_predict(predictor, ort_session, 425, 780, image, image_embedding, show=True)
    # run_predict(predictor, ort_session, 825, 850, image, image_embedding, show=True)