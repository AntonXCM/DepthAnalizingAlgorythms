import cv2
from ultralytics import YOLO
from Detector import DetectorBase

# ----Налаштування----
MODEL = YOLO("yolov8n.pt")
HUMAN_HEIGHT = 170000
# --------------------
class Detector(DetectorBase):
    def __init__(self, image):
        class Box:
            distance = 0
        def calculate_boxes(boxes):
            result = []
            for b in boxes:
                x1, y1, x2, y2 = b.xyxy[0]
                box = Box()
                box.x1 = int(x1)
                box.x2 = int(x2)
                box.y1 = int(y1)
                box.y2 = int(y2)
                box.width    = abs(box.x1 - box.x2)
                box.height   = abs(box.y1 - box.y2)
                box.distance = 0 if box.height == 0 else HUMAN_HEIGHT / box.height
                result.append(box)
            result.sort(key= lambda box : -box.distance)
            return result

        def draw_depth_map(boxes):
            x, y = image.shape[:2]
            cv2.rectangle(image, (0,0), (y,x),[0,0,0], -1)
            for box in boxes:

                cv2.rectangle(image, (box.x1, box.y1), (box.x2, box.y2), [box.distance % 255, int(box.distance / 256) % 255, int(box.distance / 65536) % 255],-1)
            cv2.imwrite("box detected.png", image)

        draw_depth_map(
            calculate_boxes(
                MODEL(image, classes=[0], verbose=False, imgsz=4096)[0].boxes
                ))
        
        self.image = image

    def getDepth(self, x, y, properties) -> str:
        r, g, b = self.image[y,x]
        return f'{((r + g * 256.0 + b * 65536.0) * 0.01 * properties["Box correct coefficient"]):.2f}'