"""Train YOLOv8 defect detection model"""
import sys
sys.path.append('grader')

from grader.defect_detector import CoffeeDefectDetector

# Initialize detector
detector = CoffeeDefectDetector()

# Train the model
print("🚀 Training defect detection model...")
results = detector.train(
    data_yaml_path='defect_dataset/data.yaml',
    epochs=15,
    imgsz=416
)

print("\n✅ Training complete!")
print(f"Model saved to: models/")