import cv2, PIL
from transformers import pipeline
from Detector import DetectorBase

# ----Налаштування----
PIPELINE = pipeline(task="depth-estimation", model="depth-anything/Depth-Anything-V2-Small-hf")
# --------------------
class Detector(DetectorBase):
    def __init__(self, image):
        image = PIPELINE(PIL.Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)))["depth"]
        image.save('depthmap.png')
        self.image = image

    def getDepth(self, x, y, properties) -> str:
        return f'{((256 - self.image.getpixel((x,y)))*properties["Depthmap correct coefficient"]):.2f}'