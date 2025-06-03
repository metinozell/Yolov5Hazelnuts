from ultralytics import YOLO
import numpy as np
import cv2

class HazelnutDetector:
    def __init__(self, model_path):
        self.model = YOLO(model_path)

    def detect(self, image):
        original_h, original_w = image.shape[:2]
        resized_image = cv2.resize(image, (640, 640))
        results = self.model(resized_image)

        class_names = []
        confidences = []
        boxes = []

        for result in results:
            if result.boxes is None:
                continue

            for box in result.boxes:
                conf = float(box.conf[0])
                if conf < 0.7:
                    continue

                cls_id = int(box.cls[0])
                class_name = self.model.names[cls_id]

                x1, y1, x2, y2 = box.xyxy[0]
                x1 = int(x1 * original_w / 640)
                x2 = int(x2 * original_w / 640)
                y1 = int(y1 * original_h / 640)
                y2 = int(y2 * original_h / 640)
                width = x2 - x1
                height = y2 - y1

                boxes.append([x1, y1, width, height])  
                confidences.append(conf)
                class_names.append(class_name)

       
        indices = cv2.dnn.NMSBoxes(boxes, confidences, score_threshold=0.7, nms_threshold=0.4)

        final_class_names = []
        final_confidences = []
        final_boxes = []

        for i in indices:
            i = i[0] if isinstance(i, (list, tuple, np.ndarray)) else i
            x, y, w, h = boxes[i]
            final_boxes.append([x, y, x + w, y + h])
            final_confidences.append(confidences[i])
            final_class_names.append(class_names[i])

        return final_class_names, final_confidences, final_boxes
