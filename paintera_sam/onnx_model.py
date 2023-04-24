import math
import time
import warnings

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


def show_mask(mask, ax):
    color = np.array([30 / 255, 144 / 255, 255 / 255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels == 1]
    neg_points = coords[labels == 0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white',
               linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white',
               linewidth=1.25)


def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0, 0, 0, 0), lw=2))

h_model = "./sam_vit_h_4b8939.pth"
h_model_type = "vit_h"
h_onnx_model = "./sam_vit_h_4b8939.onnx"
h_quantized_onnx = "./sam_vit_h_4b8939_quantized.onnx"

b_model = "./sam_vit_b_01ec64.pth"
b_model_type = "vit_b"
b_onnx_model_path = "./sam_vit_b_01ec64.onnx"
b_quantized_onnx = "./sam_vit_b_01ec64_quantized.onnx"


def export_onnx_model(sam: Sam, onnx_model_path: str, quantized_onnx_model_path: str):
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


def load_model(onnx_model: str = h_onnx_model, sam_model: str = h_model, model_type: str = h_model_type,
               device: str = "cuda") -> tuple[SamPredictor, InferenceSession]:
    model = sam_model_registry[model_type](checkpoint=sam_model)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    session = onnxruntime.InferenceSession(onnx_model, providers=['CUDAExecutionProvider'])
    model.to(device=device)
    return SamPredictor(model), session


def predict_current_image(ort_session: InferenceSession, points_in: list[list[int]], points_out: list[list[int]], image, image_embedding, threshold: float = 5.0,
                          show=False):
    # input_label = np.array([1])
    #
    # onnx_label = np.concatenate([input_label, np.array([-1])], axis=0)[None, :].astype(np.float32)

    labels = []
    show_labels = []
    show_coords = []
    coords = []
    for point in points_in:
        show_coords += [point]
        coords += [convert_coordinate(image.shape[:2], 1024, point)]
        labels += [1]
        show_labels += [1]


    for point in points_out:
        show_coords += [point]
        coords += [convert_coordinate(image.shape[:2], 1024, point)]
        labels += [0]
        show_labels += [0]

    if (len(points_out) == 0):
        labels += [0]
        coords += [[0, 0]]

    onnx_coord = np.array([coords]).astype(np.float32)
    onnx_label = np.array([labels]).astype(np.float32)

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

    masks, _, low_res_logits = ort_session.run(None, ort_inputs)
    masks = masks > threshold

    if show:
        plt.figure(figsize=(10, 10))
        plt.imshow(image)
        show_mask(masks, plt.gca())
        show_points(np.array(show_coords), np.array(show_labels), plt.gca())
        plt.axis('off')
        plt.show()

    return masks


def convert_coordinate(image_size: tuple[int, int], target_length: int, xy: tuple[int, int]) -> tuple[float, float]:
    height, width = image_size
    x, y = xy

    scale = target_length * 1.0 / max(width, height)
    scaled_width, scaled_height = int((width * scale) + 0.5), int((height * scale) + 0.5)
    new_x, new_y = x * (scaled_width / width), y * (scaled_height / height)
    return [new_x, new_y]


def predict_new_image(img: str, x: int, y: int, out="/tmp/mask.png", predictor: SamPredictor = None,
                      ort_session: onnxruntime.InferenceSession = None, show=False):
    if predictor is None or ort_session is None:
        predictor, ort_session = load_model()

    image = cv2.imread(img)
    predictor.set_image(image, image_format='BGR')
    image_embedding = predictor.get_image_embedding().cpu().numpy()

    masks = predict_current_image(ort_session, [[x,y]], [], image, image_embedding, show=show)

    h, w = masks[0].shape[-2:]
    mask_image = masks[0].reshape(h, w, 1) * 255
    cv2.imwrite(out, mask_image)

    return image, image_embedding


if __name__ == '__main__':
    model_type = h_model_type
    checkpoint = h_model

    onnx = h_onnx_model
    quantized_onnx = h_quantized_onnx
    # export_onnx_model(sam, onnx, quantized_onnx)

    begin = time.time()
    start = time.time()
    print("Load model...", end="")
    (predictor, ort_session) = load_model(quantized_onnx, checkpoint, model_type)  # quantized_onnx)
    print(time.time() - start)
    print()

    # for i in range(20):

    start = time.time()
    (image, image_embedding) = predict_new_image("/tmp/sam.png", 225, 750, predictor=predictor, ort_session=ort_session, show=False)

    predict_current_image(ort_session, [[125, 225]], [[int((225+125)/2), int((225+125)/2)]], image, image_embedding, show=True)
    predict_current_image(ort_session, [[125, 225]], [], image, image_embedding, show=True)
