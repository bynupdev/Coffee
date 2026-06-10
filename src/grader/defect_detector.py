"""
YOLOv8-based Coffee Bean Defect Detection System
Detects: good_bean, broken_bean, black_bean, moldy_bean, insect_damage, foreign_object
"""

import cv2
import numpy as np
from pathlib import Path
import json
from datetime import datetime
from collections import defaultdict
import os
import shutil

class CoffeeDefectDetector:
    """Coffee bean defect and foreign object detection using YOLOv8"""
    
    def __init__(self, model_path=None):
        self.model_path = model_path
        self.model = None
        self.classes = [
            'good_bean',
            'broken_bean', 
            'black_bean',
            'moldy_bean',
            'insect_damage',
            'foreign_object'
        ]
        
        # Color mapping for visualization
        self.class_colors = {
            'good_bean': (0, 255, 0),        # Green
            'broken_bean': (255, 255, 0),     # Yellow
            'black_bean': (0, 0, 0),          # Black
            'moldy_bean': (128, 0, 128),      # Purple
            'insect_damage': (0, 0, 255),     # Red
            'foreign_object': (255, 0, 0)     # Blue
        }
        
        # Initialize model
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize or load YOLOv8 model"""
        try:
            from ultralytics import YOLO
            
            if self.model_path and Path(self.model_path).exists():
                print(f"📦 Loading existing model from {self.model_path}")
                self.model = YOLO(self.model_path)
            else:
                print("🆕 Creating new YOLOv8n model with pretrained weights")
                self.model = YOLO('yolov8n.pt')  # Nano version for speed
                
        except ImportError:
            print("⚠️ Ultralytics not installed. Using OpenCV-based detection fallback.")
            self.model = None
    
    def create_dataset_structure(self, base_dir='defect_dataset'):
        """Create YOLO format dataset structure"""
        base_path = Path(base_dir)
        
        # Create directories
        for split in ['train', 'val', 'test']:
            (base_path / split / 'images').mkdir(parents=True, exist_ok=True)
            (base_path / split / 'labels').mkdir(parents=True, exist_ok=True)
        
        # Create data.yaml
        yaml_content = {
            'path': str(base_path.absolute()),
            'train': 'train/images',
            'val': 'val/images',
            'test': 'test/images',
            'nc': len(self.classes),
            'names': self.classes
        }
        
        import yaml
        with open(base_path / 'data.yaml', 'w') as f:
            yaml.dump(yaml_content, f, default_flow_style=False)
        
        print(f"✅ Dataset structure created at {base_path}")
        return base_path
    
    def label_image_with_cv(self, image_path, output_dir=None):
        """
        Use OpenCV to automatically detect and label defects
        This serves as auto-labeling for training data
        """
        img = cv2.imread(str(image_path))
        if img is None:
            return None
        
        h, w = img.shape[:2]
        labels = []
        
        # Convert to different color spaces for analysis
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Detect beans using contour analysis
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < 100:  # Filter noise
                continue
            
            x, y, bw, bh = cv2.boundingRect(contour)
            
            # Normalize coordinates for YOLO format
            x_center = (x + bw/2) / w
            y_center = (y + bh/2) / h
            width = bw / w
            height = bh / h
            
            # Analyze the bean region
            bean_roi = img[y:y+bh, x:x+bw]
            bean_hsv = hsv[y:y+bh, x:x+bw]
            
            if bean_roi.size == 0:
                continue
            
            # Classify the bean/object
            bean_class = self._classify_object(bean_roi, bean_hsv, area, bw, bh)
            
            labels.append({
                'class': self.classes.index(bean_class),
                'class_name': bean_class,
                'bbox': [x_center, y_center, width, height],
                'conf': 1.0  # Auto-labeled
            })
        
        # Save labels if output directory provided
        if output_dir and labels:
            label_path = Path(output_dir) / f"{Path(image_path).stem}.txt"
            with open(label_path, 'w') as f:
                for label in labels:
                    bbox = label['bbox']
                    f.write(f"{label['class']} {bbox[0]:.6f} {bbox[1]:.6f} {bbox[2]:.6f} {bbox[3]:.6f}\n")
        
        return labels
    
    def _classify_object(self, roi, roi_hsv, area, width, height):
        """Classify detected object based on visual features"""
        
        # Calculate features
        mean_color = np.mean(roi, axis=(0, 1))
        mean_hsv = np.mean(roi_hsv, axis=(0, 1))
        aspect_ratio = width / height if height > 0 else 0
        
        # Color-based classification
        # Black beans (very dark)
        if mean_hsv[2] < 50:
            return 'black_bean'
        
        # Moldy beans (green/blue tint)
        if 40 < mean_hsv[0] < 85 and mean_hsv[1] > 30:
            return 'moldy_bean'
        
        # Foreign objects (non-brown colors)
        if mean_hsv[0] < 10 or mean_hsv[0] > 30:
            if area > 5000:  # Large objects
                return 'foreign_object'
        
        # Broken beans (irregular shape)
        if abs(aspect_ratio - 1.0) > 0.5:
            return 'broken_bean'
        
        # Insect damage (small holes/irregularities)
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / (width * height)
        if edge_density > 0.3:
            return 'insect_damage'
        
        # Default to good bean
        return 'good_bean'
    
    def train(self, data_yaml_path, epochs=15, imgsz=416):
        """Train the YOLOv8 model"""
        if self.model is None:
            print("❌ Model not initialized")
            return
        
        print(f"🚀 Starting training for {epochs} epochs...")
        
        # Train the model
        results = self.model.train(
            data=data_yaml_path,
            epochs=epochs,
            imgsz=imgsz,
            batch=16,
            device='cpu',  # CPU training
            workers=4,
            patience=5,
            save=True,
            save_period=5
        )
        
        # Save the trained model
        model_save_path = Path('models') / f'coffee_defect_detector_{datetime.now().strftime("%Y%m%d_%H%M%S")}.pt'
        self.model.save(str(model_save_path))
        self.model_path = str(model_save_path)
        
        print(f"✅ Training complete! Model saved to {model_save_path}")
        return results
    
    def detect(self, image_path, conf_threshold=0.25):
        """
        Detect defects in a single image
        Returns detections with bounding boxes
        """
        if self.model is None:
            return self._fallback_detect(image_path)
        
        # Run YOLOv8 inference
        results = self.model(image_path, conf=conf_threshold, device='cpu')
        
        detections = []
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    conf = box.conf[0].item()
                    cls = int(box.cls[0].item())
                    
                    detections.append({
                        'class': self.classes[cls],
                        'confidence': round(conf, 4),
                        'bbox': [round(x1, 2), round(y1, 2), round(x2, 2), round(y2, 2)]
                    })
        
        return detections
    
    def _fallback_detect(self, image_path):
        """Fallback detection using OpenCV when YOLO is not available"""
        labels = self.label_image_with_cv(image_path)
        
        if labels is None:
            return []
        
        detections = []
        img = cv2.imread(str(image_path))
        h, w = img.shape[:2]
        
        for label in labels:
            x_center, y_center, width, height = label['bbox']
            x1 = (x_center - width/2) * w
            y1 = (y_center - height/2) * h
            x2 = (x_center + width/2) * w
            y2 = (y_center + height/2) * h
            
            detections.append({
                'class': label['class_name'],
                'confidence': 0.85,  # Estimated confidence
                'bbox': [round(x1, 2), round(y1, 2), round(x2, 2), round(y2, 2)]
            })
        
        return detections
    
    def analyze_image(self, image_path, conf_threshold=0.25):
        """
        Analyze single image for defects and compute statistics
        """
        # Run detection
        detections = self.detect(image_path, conf_threshold)
        
        # Count objects by class
        counts = defaultdict(int)
        for det in detections:
            counts[det['class']] += 1
        
        # Calculate statistics
        total_beans = sum([
            counts['good_bean'],
            counts['broken_bean'],
            counts['black_bean'],
            counts['moldy_bean'],
            counts['insect_damage']
        ])
        
        total_defects = sum([
            counts['broken_bean'],
            counts['black_bean'],
            counts['moldy_bean'],
            counts['insect_damage']
        ])
        
        impurity_count = counts['foreign_object']
        
        # Compute ratios
        defect_ratio = total_defects / total_beans if total_beans > 0 else 0
        impurity_ratio = impurity_count / (total_beans + impurity_count) if (total_beans + impurity_count) > 0 else 0
        
        # Quality score based on good beans percentage and defect types
        good_bean_ratio = counts['good_bean'] / total_beans if total_beans > 0 else 0
        quality_score = max(0, min(100, good_bean_ratio * 100))
        
        return {
            'detections': detections,
            'counts': dict(counts),
            'total_beans': total_beans,
            'total_defects': total_defects,
            'impurity_count': impurity_count,
            'defect_ratio': round(defect_ratio, 4),
            'impurity_ratio': round(impurity_ratio, 4),
            'quality_score': round(quality_score, 2)
        }
    
    def analyze_sample(self, image_paths, conf_threshold=0.25):
        """
        Analyze multiple images as a sample batch
        Returns aggregated statistics
        """
        image_results = []
        
        # Analyze each image
        for img_path in image_paths:
            if Path(img_path).exists():
                result = self.analyze_image(img_path, conf_threshold)
                image_results.append({
                    'image_path': str(img_path),
                    'result': result
                })
        
        # Aggregate statistics
        if not image_results:
            return {'error': 'No valid images analyzed'}
        
        num_images = len(image_results)
        
        # Initialize aggregators
        avg_counts = defaultdict(float)
        avg_defect_ratio = 0
        avg_impurity_ratio = 0
        avg_quality_score = 0
        
        for img_result in image_results:
            result = img_result['result']
            
            for cls in self.classes:
                avg_counts[cls] += result['counts'].get(cls, 0)
            
            avg_defect_ratio += result['defect_ratio']
            avg_impurity_ratio += result['impurity_ratio']
            avg_quality_score += result['quality_score']
        
        # Calculate averages
        for cls in self.classes:
            avg_counts[cls] = round(avg_counts[cls] / num_images, 2)
        
        avg_defect_ratio = round(avg_defect_ratio / num_images, 4)
        avg_impurity_ratio = round(avg_impurity_ratio / num_images, 4)
        avg_quality_score = round(avg_quality_score / num_images, 2)
        
        return {
            'sample_summary': {
                'num_images': num_images,
                'average_defect_ratio': avg_defect_ratio,
                'average_impurity_ratio': avg_impurity_ratio,
                'average_quality_score': avg_quality_score,
                'average_counts': dict(avg_counts)
            },
            'image_results': image_results
        }
    
    def visualize_detections(self, image_path, output_path=None):
        """
        Draw bounding boxes on image with class labels
        """
        img = cv2.imread(str(image_path))
        if img is None:
            return None
        
        result = self.analyze_image(image_path)
        
        for det in result['detections']:
            x1, y1, x2, y2 = [int(c) for c in det['bbox']]
            cls_name = det['class']
            conf = det['confidence']
            
            color = self.class_colors.get(cls_name, (255, 255, 255))
            
            # Draw bounding box
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            
            # Draw label
            label = f"{cls_name} {conf:.2f}"
            (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(img, (x1, y1 - label_h - 10), (x1 + label_w, y1), color, -1)
            cv2.putText(img, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Add summary statistics
        summary_lines = [
            f"Total Beans: {result['total_beans']}",
            f"Defect Ratio: {result['defect_ratio']:.2%}",
            f"Quality Score: {result['quality_score']:.1f}%",
            f"Impurities: {result['impurity_count']}"
        ]
        
        y_offset = 30
        for line in summary_lines:
            cv2.putText(img, line, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            y_offset += 25
        
        if output_path:
            cv2.imwrite(str(output_path), img)
        
        return img
    
    def prepare_training_data(self, source_images_dir, output_dataset_dir='defect_dataset'):
        """
        Auto-label images for training using OpenCV-based detection
        """
        from sklearn.model_selection import train_test_split
        
        dataset_path = self.create_dataset_structure(output_dataset_dir)
        source_path = Path(source_images_dir)
        
        # Collect all images
        image_files = []
        for ext in ['*.jpg', '*.jpeg', '*.png']:
            image_files.extend(source_path.glob(ext))
        
        if not image_files:
            print("❌ No images found in source directory")
            return
        
        # Split into train/val/test
        train_files, temp_files = train_test_split(image_files, test_size=0.3, random_state=42)
        val_files, test_files = train_test_split(temp_files, test_size=0.5, random_state=42)
        
        splits = {
            'train': train_files,
            'val': val_files,
            'test': test_files
        }
        
        # Process each split
        for split_name, files in splits.items():
            img_dir = dataset_path / split_name / 'images'
            lbl_dir = dataset_path / split_name / 'labels'
            
            for img_file in files:
                # Copy image
                shutil.copy(img_file, img_dir / img_file.name)
                
                # Generate labels
                self.label_image_with_cv(str(img_file), str(lbl_dir))
        
        print(f"✅ Training data prepared:")
        print(f"   Train: {len(train_files)} images")
        print(f"   Val: {len(val_files)} images")
        print(f"   Test: {len(test_files)} images")
        
        return dataset_path / 'data.yaml'

# Global detector instance
detector = CoffeeDefectDetector()