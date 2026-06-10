"""Auto-label and prepare defect detection dataset"""
import sys
sys.path.append('grader')

from grader.defect_detector import CoffeeDefectDetector
from pathlib import Path

# Initialize detector
detector = CoffeeDefectDetector()

# Source images directory (your organized images)
source_dir = Path('defect_dataset_source')

# Prepare training dataset
print("🔍 Auto-labeling images and preparing dataset...")
data_yaml = detector.prepare_training_data(
    source_images_dir=source_dir,
    output_dataset_dir='defect_dataset'
)

print(f"\n✅ Dataset prepared!")
print(f"📁 Data YAML: {data_yaml}")
print("\nNext step: Train the model using:")
print("python train_defect_model.py")