from ultralytics import YOLO

# Load model
model = YOLO("best.onnx")

# Run inference
results = model(["test_findik.jpg"])

# Process results
for result in results:
    boxes = result.boxes
    class_ids = boxes.cls  # Tensor of predicted class IDs
    confidences = boxes.conf  # Tensor of confidence scores

    for class_id, conf in zip(class_ids, confidences):
        label = model.names[int(class_id)]  # Convert class ID to label name
        print(f"Predicted label: {label}, Confidence: {conf:.2f}")
