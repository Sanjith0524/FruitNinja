from ultralytics import YOLO
import torch
import os

def setup_gpu_training():
    if torch.cuda.is_available():
        device = torch.cuda.current_device()
        print(f"GPU is available! Using {torch.cuda.get_device_name(device)}")
        gpu_memory = torch.cuda.get_device_properties(device).total_memory / 1024**3
        print(f"Total GPU memory: {gpu_memory:.2f} GB")
        print(f"GPU Count: {torch.cuda.device_count()}")
        torch.cuda.set_device(device)
        return True
    else:
        print("No GPU found. Please check your CUDA installation.")
        return False

def train_model(data_yaml_path, epochs=50, batch_size=8, img_size=640, model_type="yolov8s.pt"):
    try:
        model = YOLO(model_type)
        training_args = {
            "data": data_yaml_path,
            "epochs": epochs,
            "imgsz": img_size,
            "batch": batch_size,
            "name": "orange_classifier_rtx4050",
            "device": 0,
            "workers": 4,
            "patience": 50,
            "save": True,
            "cache": False,
            "verbose": True,
            "amp": False,
            "optimizer": "SGD",
            "lr0": 0.01,
            "lrf": 0.01,
            "momentum": 0.937,
            "weight_decay": 0.0005,
            "warmup_epochs": 3.0,
            "warmup_momentum": 0.8,
            "warmup_bias_lr": 0.1,
            "box": 7.5,
            "cls": 0.5,
            "dfl": 1.5,
            "close_mosaic": 10,
            "val": True,
            "rect": False,
            "mixup": 0.0,
            "mosaic": 0.0,
            "degrees": 0.0,
            "translate": 0.1,
            "scale": 0.5,
            "shear": 0.0,
            "perspective": 0.0,
            "flipud": 0.0,
            "fliplr": 0.5,
            "copy_paste": 0.0,
        }
        results = model.train(**training_args)
        print("Training completed successfully!")
        model_path = os.path.join('runs', 'detect', 'orange_classifier_rtx4050', 'weights', 'best.pt')
        if os.path.exists(model_path):
            print(f"Best model saved at: {model_path}")
        return model
    except Exception as e:
        print(f"Training failed with error: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    data_yaml_path = "D:/Fruit Ninja/data.yaml"
    if setup_gpu_training():
        print("Starting training with simplified configuration...")
        trained_model = train_model(
            data_yaml_path=data_yaml_path,
            epochs=50,
            batch_size=8,
            img_size=640,
            model_type="yolov8s.pt"
        )
        if trained_model:
            print("Training successful!")
            try:
                trained_model.export(format="onnx")
                print("Model exported successfully!")
            except Exception as e:
                print(f"Export failed: {str(e)}")
