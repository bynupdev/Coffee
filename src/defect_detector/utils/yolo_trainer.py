"""
YOLO model training for defect detection
"""
import os
import json
from pathlib import Path
from datetime import datetime
from ultralytics import YOLO

class YOLOTrainer:
    """Trains YOLO models for coffee bean defect detection"""
    
    def __init__(self, model_size='n'):
        """
        Args:
            model_size: 'n' (nano), 's' (small), 'm' (medium), 'l' (large)
        """
        self.model_size = model_size
        self.model = None
        self.results = None
        
    def train(self, data_yaml, epochs=50, imgsz=416, batch=16, 
             project_name='coffee_defects', custom_weight=1.0):
        """
        Train the YOLO model
        
        Args:
            data_yaml: Path to data.yaml file
            epochs: Number of training epochs
            imgsz: Image size for training
            batch: Batch size
            project_name: Name for the training run
            custom_weight: Reserved for future custom data integration
        """
        print(f"Loading YOLOv8{self.model_size} model...")
        self.model = YOLO(f'yolov8{self.model_size}.pt')
        
        # Training configuration
        results = self.model.train(
            data=data_yaml,
            epochs=epochs,
            imgsz=imgsz,
            batch=batch,
            device='cpu',  # Change to 'cuda' if GPU available
            workers=4,
            patience=10,
            save=True,
            save_period=10,
            project=project_name,
            name=f"train_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            exist_ok=True,
            pretrained=True,
            optimizer='Adam',
            lr0=0.001,
            cos_lr=True,
            augment=True
        )
        
        self.results = results
        return results
    
    def evaluate(self):
        """Evaluate the trained model"""
        if self.model is None:
            raise ValueError("No model trained yet")
        
        metrics = self.model.val()
        return {
            'mAP50': metrics.box.map50,
            'mAP50_95': metrics.box.map,
            'precision': metrics.box.p[0] if metrics.box.p is not None else 0,
            'recall': metrics.box.r[0] if metrics.box.r is not None else 0,
        }
    
    def predict_on_image(self, image_path, conf_threshold=0.25):
        """Run inference on a single image"""
        if self.model is None:
            raise ValueError("No model trained yet")
        
        results = self.model(image_path, conf=conf_threshold)
        
        detections = []
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    conf = box.conf[0].item()
                    cls = int(box.cls[0].item())
                    
                    detections.append({
                        'class_id': cls,
                        'class_name': self.model.names[cls],
                        'confidence': round(conf, 4),
                        'bbox': [round(x1), round(y1), round(x2), round(y2)]
                    })
        
        return detections
    
    def save_model(self, save_path):
        """Save the trained model"""
        if self.model is None:
            raise ValueError("No model to save")
        self.model.save(save_path)