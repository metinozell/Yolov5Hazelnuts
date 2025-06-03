import os
import cv2
import serial
import requests
import numpy as np
import time
from detector import HazelnutDetector
from fastapi import FastAPI
from threading import Thread
import uvicorn
from pydantic import BaseModel
from dotenv import load_dotenv,dotenv_values

class HazelnutSorter:
    def __init__(self, model_path, serial_port='COM4'):
        self.servo_bad_count = 0
        self.servo_quality_count = 0
        self.model_path = model_path
        self.detector = HazelnutDetector(self.model_path)
        self.camera_url = os.getenv("MY_CAMERA_URL") 
        
        try:
            self.arduino = serial.Serial(serial_port, 115200, timeout=1)
            time.sleep(2)
        except serial.SerialException as e:
            print(f"Arduino bağlantı hatası: {e}")
            self.arduino = None

        self.start_api_server()

    def get_camera_frame(self):
        try:
            response = requests.get(self.camera_url, timeout=5)
            if response.status_code == 200:
                img_array = np.frombuffer(response.content, np.uint8)
                frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
                return frame
        except requests.RequestException as e:
            print(f"Görüntü alma hatası: {e}")
        return None

    def process_hazelnut(self):
        frame = self.get_camera_frame()
        
        if frame is None:
            print("Görüntü çekilemedi.")
            return
        frame = cv2.resize(frame, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_LINEAR)

        class_names, confidences, boxes = self.detector.detect(frame)

        detected_any = False

        for i, (box, conf) in enumerate(zip(boxes, confidences)):
            x1, y1, x2, y2 = box
            label = f"{class_names[i]} {conf*100:.1f}%"
            color = (0, 255, 0) if "qualityHazelnut" in class_names[i] else (0, 0, 255)
            print(f"Hazelnut Detected as {label} with confidence {conf}")

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            detected_any = True

        cv2.imshow("Hazelnut Detection", frame)
        cv2.waitKey(1)

        if detected_any and self.arduino:
            if "qualityHazelnut" in class_names:
                self.servo_quality_count += 1
                self.arduino.write(b'GOOD\n')
            elif "damagedHazelnut" in class_names:
                self.servo_bad_count += 1
                self.arduino.write(b'BAD\n')
            self.arduino.flush()

        else:
            print("Geçerli tespit yok, Arduino'ya sinyal gönderilmedi.")

    def start_api_server(self):
        app = FastAPI()

        class ServoResponse(BaseModel):
            servo_bad_count: int
            servo_quality_count: int

        @app.get("/servo_counter", response_model=ServoResponse)
        async def get_count():
            return ServoResponse(servo_bad_count=self.servo_bad_count, servo_quality_count=self.servo_quality_count)

        def run():
            uvicorn.run(app, host=os.getenv("MY_HOST"), port=os.getenv("MY_PORT"))

        Thread(target=run, daemon=True).start()

    def run(self):
        while True:
            self.process_hazelnut()
            time.sleep(3)
