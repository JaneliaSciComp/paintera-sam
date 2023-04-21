import sys

import cv2

from paintera_sam import onnx_model


class OnnxPainteraPredictor:
    sam_predictor, ort_session = onnx_model.load_model()
    image = None
    image_embedding = None
    mask = None

    def set_image(self, img, image_format="BGR"):
        if isinstance(img, str):
            OnnxPainteraPredictor.image = cv2.imread(img)
        else:
            OnnxPainteraPredictor.image = img

        OnnxPainteraPredictor.sam_predictor.set_image(OnnxPainteraPredictor.image, image_format)
        OnnxPainteraPredictor.image_embedding = OnnxPainteraPredictor.sam_predictor.get_image_embedding().cpu().numpy()

    def predict(self, points_in, points_out, threshold: float = 5.0, show=False):
        masks = onnx_model.predict_current_image(OnnxPainteraPredictor.ort_session, points_in, points_out,
                                                 OnnxPainteraPredictor.image,
                                                 OnnxPainteraPredictor.image_embedding, threshold, show)
        OnnxPainteraPredictor.mask = masks[0]
        return OnnxPainteraPredictor.mask

    def save_mask(self, out="/tmp/mask.png"):

        mask = OnnxPainteraPredictor.mask
        if mask is not None:
            h, w = mask.shape[-2:]
            mask_image = mask.reshape(h, w, 1) * 255
            cv2.imwrite(out, mask_image)


onnx_predictor = OnnxPainteraPredictor()


def set_image(img, image_format="BGR"):
    onnx_predictor.set_image(img, image_format)


def predict(points_in, points_out, threshold: float = 5.0, show=False):
    onnx_predictor.predict(points_in, points_out, threshold, show)


def save_mask(out="/tmp/mask.png"):
    onnx_predictor.save_mask(out)


if __name__ == "__main__":
    set_image("/tmp/sam.png")

    print()
    print("Image Predication Time:")
    import time

    for i in range(3):
        print(f"{i}\t", end="")
        start = time.time()
        predict([[300, 300]], [])
        print(f"{time.time() - start}")
    predict([[300, 300]], [], True)
    predict([[200, 600]], [], True)
    predict([[600, 200]], [], True)

    save_mask()
