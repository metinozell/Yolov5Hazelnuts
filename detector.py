from ultralytics import YOLO
import numpy as np
import cv2  # resize ve görüntü işlemede gerekli

class HazelnutDetector:
    def __init__(self, model_path):
        self.model = YOLO(model_path)

    def detect(self, image):
        original_h, original_w = image.shape[:2]  # Orijinal boyutları al
        resized_image = cv2.resize(image, (640, 640))  # 640x640 boyutuna getir

        results = self.model(resized_image)  # Modeli bu boyutta çalıştır

        class_names = []
        confidences = []
        boxes = []

        for result in results:
            if result.boxes is None:
                continue  # Tespit yoksa atla

            for box in result.boxes:
                conf = float(box.conf[0])
                if conf < 0.5:
                    continue  # Güven eşiği

                cls_id = int(box.cls[0])
                class_name = self.model.names[cls_id]
                class_names.append(class_name)
                confidences.append(conf)

                # Box koordinatlarını al ve orijinal boyuta geri ölçekle
                x1, y1, x2, y2 = box.xyxy[0]
                x1 = int(x1 * original_w / 640)
                x2 = int(x2 * original_w / 640)
                y1 = int(y1 * original_h / 640)
                y2 = int(y2 * original_h / 640)
                boxes.append([x1, y1, x2, y2])

            print("Boxes:", boxes)

        return class_names, confidences, boxes
