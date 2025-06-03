from ultralytics import YOLO

if __name__ == "__main__":
    model = YOLO("yolo11n.pt")

    
    train_results = model.train(
        data="data_v4\data.yaml",  
        epochs=50,  
        imgsz=640,  
        device=0,  
    )

   
    metrics = model.val()

    path = model.export(format="onnx") 