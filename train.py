from ultralytics import YOLO

if __name__ == "__main__":
    # Load a pretrained YOLO11n model
    model = YOLO("yolo11n.pt")

    # Train the model on the COCO8 dataset for 100 epochs
    train_results = model.train(
        data="data_v3\data.yaml",  # Path to dataset configuration file
        epochs=50,  # Number of training epochs
        imgsz=640,  # Image size for training
        device=0,  # Device to run on (e.g., 'cpu', 0, [0,1,2,3])
    )

    # Evaluate the model's performance on the validation set
    metrics = model.val()

    # Perform object detection on an image
    # results = model("data_v2\test\images\damaged_nut_00006_jpg.rf.da7809de63736791addf9413ec7deb7e.jpg")  # Predict on an image
    # results[0].show()  # Display results

    # Export the model to ONNX format for deployment
    path = model.export(format="onnx")  # Returns the path to the exported model

    # OLD: python train.py --img 640 --batch 16 --epochs 50 --data data/data.yaml --weights yolov5s.pt device 0