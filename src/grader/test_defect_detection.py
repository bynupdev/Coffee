"""Test defect detection on sample images"""
import sys
sys.path.append('grader')

from grader.defect_detector import CoffeeDefectDetector
from grader.prediction import IntegratedCoffeeAnalyzer
import json

# Initialize systems
detector = CoffeeDefectDetector()
analyzer = IntegratedCoffeeAnalyzer()

# Test single image
test_image = "test_coffee_beans.jpg"
print(f"\n🔍 Analyzing: {test_image}")

# Defect detection only
print("\n1. Defect Detection Results:")
result = detector.analyze_image(test_image)
print(json.dumps(result, indent=2))

# Visualize detections
vis_path = "visualized_test.jpg"
detector.visualize_detections(test_image, vis_path)
print(f"\n📸 Visualization saved to: {vis_path}")

# Full analysis
print("\n2. Complete Analysis:")
full_result = analyzer.analyze_single_image(test_image)
print(json.dumps(full_result, indent=2))

# Batch analysis
print("\n3. Batch Sample Analysis:")
sample_images = ["bean1.jpg", "bean2.jpg", "bean3.jpg"]
batch_result = analyzer.analyze_sample_batch(sample_images)
print(json.dumps(batch_result, indent=2))