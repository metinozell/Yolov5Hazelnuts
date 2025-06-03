from ultralytics import YOLO


model = YOLO("best.onnx")


results = model(["test.jpg"])


for result in results:
    boxes = result.boxes
    class_ids = boxes.cls  
    confidences = boxes.conf  

    for class_id, conf in zip(class_ids, confidences):
        label = model.names[int(class_id)]  
        print(f"Predicted label: {label}, Confidence: {conf:.2f}")
