import sys

import cv2

import paintera_sam.sam_model as model

class PainteraPredictor:

    sam_predictor = model.new_predictor()
    image = None
    mask = None

    def __init__(self):
        pass
    def set_image(self, img, image_format="BGR"):
        if isinstance(img, str):
            PainteraPredictor.image = cv2.imread(img)
        else:
            PainteraPredictor.image = img

        PainteraPredictor.sam_predictor.set_image(PainteraPredictor.image, image_format)

    def predict(self, x, y, show=False):
        if isinstance(x, int) and isinstance(y, int):
            masks = model.predict_current_image(PainteraPredictor.sam_predictor, x, y, PainteraPredictor.image, show)
            PainteraPredictor.mask = masks[0]
        else:
            pass
        return PainteraPredictor.mask

    def save_mask(self, out="/tmp/mask.png"):

        if PainteraPredictor.mask is not None:
            cv2.imwrite(out, PainteraPredictor.mask * 255)

    def reset(self):
        pass


if __name__ == "__main__":
    paintera_sam = PainteraPredictor()
    paintera_sam.set_image("/tmp/sam.png")
    paintera_sam.predict(300, 300)
    paintera_sam.save_mask()