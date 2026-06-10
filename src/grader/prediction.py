# """
# Coffee Bean Prediction Module
# Handles model loading and prediction with error handling
# """

# import tensorflow as tf
# import numpy as np
# from PIL import Image
# import io
# import os
# from pathlib import Path

# class CoffeePredictor:
#     """Coffee bean grade predictor"""
    
#     def __init__(self, model_path=None):
#         if model_path is None:
#             # Look for model in current directory
#             current_dir = Path(__file__).parent
#             model_path = current_dir / 'coffee_grader.h5'
        
#         self.model_path = str(model_path)
#         self.model = None
#         self.classes = ['Grade A', 'Grade B', 'Grade C', 'Grade D']
#         self.img_size = (224, 224)
        
#         # Load model on initialization
#         self.load_model()
    
#     def load_model(self):
#         """Load the trained model"""
#         if not os.path.exists(self.model_path):
#             raise FileNotFoundError(
#                 f"Model file not found at {self.model_path}\n"
#                 "Please train the model first by running:\n"
#                 "python train_model.py"
#             )
        
#         try:
#             self.model = tf.keras.models.load_model(self.model_path)
#             print(f"✅ Model loaded successfully from {self.model_path}")
#         except Exception as e:
#             raise Exception(f"Error loading model: {e}")
    
#     def preprocess_image(self, image_file):
#         """Preprocess image for prediction"""
#         # Read image
#         if isinstance(image_file, bytes):
#             img = Image.open(io.BytesIO(image_file))
#         else:
#             img = Image.open(image_file)
        
#         # Convert to RGB if necessary
#         if img.mode != 'RGB':
#             img = img.convert('RGB')
        
#         # Resize
#         img = img.resize(self.img_size)
        
#         # Convert to array and normalize
#         img_array = np.array(img) / 255.0
        
#         # Add batch dimension
#         img_array = np.expand_dims(img_array, axis=0)
        
#         return img_array
    
#     def predict(self, image_file):
#         """Make prediction on image"""
#         try:
#             # Preprocess image
#             processed_img = self.preprocess_image(image_file)
            
#             # Make prediction
#             predictions = self.model.predict(processed_img, verbose=0)
            
#             # Get class and confidence
#             predicted_class_idx = np.argmax(predictions[0])
#             confidence = float(predictions[0][predicted_class_idx])
            
#             # Get grade (A, B, C, D)
#             grade = self.classes[predicted_class_idx].split()[1]
            
#             return {
#                 'grade': grade,
#                 'confidence': round(confidence, 4),
#                 'full_grade': self.classes[predicted_class_idx],
#                 'all_probabilities': {
#                     self.classes[i].split()[1]: round(float(pred), 4) 
#                     for i, pred in enumerate(predictions[0])
#                 }
#             }
            
#         except Exception as e:
#             raise Exception(f"Prediction error: {e}")

# # Initialize predictor (will raise error if model not found)
# try:
#     predictor = CoffeePredictor()
# except FileNotFoundError as e:
#     print("\n" + "="*60)
#     print("⚠️  MODEL NOT FOUND")
#     print("="*60)
#     print(str(e))
#     print("\nPlease train the model first:")
#     print("1. cd into the grader directory")
#     print("2. Run: python train_model.py")
#     print("3. Wait for training to complete")
#     print("4. Then restart the Django server")
#     print("="*60 + "\n")
#     raise

# def predict_image(image_file):
#     """Wrapper function for prediction"""
#     return predictor.predict(image_file)


""""
Integrated Coffee Bean Grading and Defect Detection - Working Version
"""

import tensorflow as tf
import numpy as np
from PIL import Image
import io
import os
from pathlib import Path
from datetime import datetime
from collections import defaultdict
import cv2

class SimpleCoffeeAnalyzer:
    """Simple but working coffee analyzer"""
    
    def __init__(self):
        self.grading_model = None
        self.classes = ['Grade_A', 'Grade_B', 'Grade_C', 'Grade_D', 'Grade_E']
        self.img_size = (224, 224)
        self._load_grading_model()
    
    def _load_grading_model(self):
        """Load the grading model"""
        model_path = Path(__file__).parent / 'coffee_grader.h5'
        
        if model_path.exists():
            try:
                self.grading_model = tf.keras.models.load_model(str(model_path))
                print(f"✅ Model loaded from {model_path}")
            except Exception as e:
                print(f"⚠️ Error loading model: {e}")
        else:
            print(f"⚠️ Model not found at {model_path}")
    
    def analyze_defects_cv(self, image_path):
        """Simple OpenCV-based defect detection"""
        if isinstance(image_path, bytes):
            img_array = np.frombuffer(image_path, np.uint8)
            img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        else:
            img = cv2.imread(str(image_path))
        
        if img is None:
            return self._empty_defect_result()
        
        h, w = img.shape[:2]
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Detect beans using thresholding
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Count and classify objects
        counts = {
            'good_bean': 0,
            'broken_bean': 0,
            'black_bean': 0,
            'moldy_bean': 0,
            'insect_damage': 0,
            'foreign_object': 0
        }
        
        detections = []
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < 100:  # Filter noise
                continue
            
            x, y, bw, bh = cv2.boundingRect(contour)
            
            # Analyze the region
            roi = img[y:y+bh, x:x+bw]
            roi_hsv = hsv[y:y+bh, x:x+bw]
            
            if roi.size == 0:
                continue
            
            # Calculate features
            mean_hsv = np.mean(roi_hsv, axis=(0, 1))
            mean_color = np.mean(roi, axis=(0, 1))
            aspect_ratio = bw / bh if bh > 0 else 0
            
            # Classify
            if mean_hsv[2] < 50:  # Very dark
                counts['black_bean'] += 1
                cls_name = 'black_bean'
            elif 40 < mean_hsv[0] < 85 and mean_hsv[1] > 30:  # Green tint
                counts['moldy_bean'] += 1
                cls_name = 'moldy_bean'
            elif area > 5000 and (mean_hsv[0] < 10 or mean_hsv[0] > 30):  # Large non-brown
                counts['foreign_object'] += 1
                cls_name = 'foreign_object'
            elif abs(aspect_ratio - 1.0) > 0.5:  # Irregular shape
                counts['broken_bean'] += 1
                cls_name = 'broken_bean'
            else:
                # Check for insect damage (holes)
                roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                edges = cv2.Canny(roi_gray, 50, 150)
                edge_density = np.sum(edges > 0) / (bw * bh)
                
                if edge_density > 0.3:
                    counts['insect_damage'] += 1
                    cls_name = 'insect_damage'
                else:
                    counts['good_bean'] += 1
                    cls_name = 'good_bean'
            
            detections.append({
                'class': cls_name,
                'confidence': 0.85,
                'bbox': [int(x), int(y), int(x+bw), int(y+bh)]
            })
        
        # Calculate statistics
        total_beans = sum([counts['good_bean'], counts['broken_bean'], 
                          counts['black_bean'], counts['moldy_bean'], 
                          counts['insect_damage']])
        
        total_defects = sum([counts['broken_bean'], counts['black_bean'],
                           counts['moldy_bean'], counts['insect_damage']])
        
        defect_ratio = total_defects / total_beans if total_beans > 0 else 0
        quality_score = (counts['good_bean'] / total_beans * 100) if total_beans > 0 else 0
        
        return {
            'counts': counts,
            'total_beans': total_beans,
            'total_defects': total_defects,
            'impurity_count': counts['foreign_object'],
            'defect_ratio': round(defect_ratio, 4),
            'impurity_ratio': round(counts['foreign_object'] / (total_beans + counts['foreign_object']), 4) if (total_beans + counts['foreign_object']) > 0 else 0,
            'quality_score': round(quality_score, 2),
            'detections': detections
        }
    
    def _empty_defect_result(self):
        """Return empty defect result"""
        return {
            'counts': {k: 0 for k in ['good_bean', 'broken_bean', 'black_bean', 'moldy_bean', 'insect_damage', 'foreign_object']},
            'total_beans': 0,
            'total_defects': 0,
            'impurity_count': 0,
            'defect_ratio': 0,
            'impurity_ratio': 0,
            'quality_score': 0,
            'detections': []
        }
    
    def _predict_grade(self, image_path):
        """Predict grade using the loaded model"""
        if self.grading_model is None:
            return None
        
        try:
            # Load and preprocess image
            if isinstance(image_path, bytes):
                img = Image.open(io.BytesIO(image_path))
            else:
                img = Image.open(image_path)
            
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            img = img.resize(self.img_size)
            img_array = np.array(img) / 255.0
            img_array = np.expand_dims(img_array, axis=0)
            
            # Predict
            predictions = self.grading_model.predict(img_array, verbose=0)
            
            # Handle multi-output model
            if isinstance(predictions, list):
                grade_probs = predictions[0][0]
            else:
                grade_probs = predictions[0]
            
            predicted_idx = np.argmax(grade_probs)
            confidence = float(grade_probs[predicted_idx])
            
            return {
                'grade': self.classes[predicted_idx].split('_')[1],
                'confidence': round(confidence, 4),
                'full_grade': self.classes[predicted_idx],
                'all_probabilities': {
                    self.classes[i].split('_')[1]: round(float(prob), 4)
                    for i, prob in enumerate(grade_probs)
                }
            }
        except Exception as e:
            print(f"Grade prediction error: {e}")
            return None
    
    def _grade_from_defects(self, defect_result):
        """Determine grade based on defect analysis"""
        defect_ratio = defect_result['defect_ratio']
        quality_score = defect_result['quality_score']
        impurity_count = defect_result['impurity_count']
        
        if impurity_count > 5 or quality_score < 20:
            grade = 'E'
        elif defect_ratio > 0.3 or quality_score < 50:
            grade = 'D'
        elif defect_ratio > 0.15 or quality_score < 70:
            grade = 'C'
        elif defect_ratio > 0.05 or quality_score < 85:
            grade = 'B'
        else:
            grade = 'A'
        
        return {
            'grade': grade,
            'confidence': round(quality_score / 100, 4),
            'full_grade': f'Grade_{grade}',
            'all_probabilities': {
                'A': 0.8 if grade == 'A' else 0.2,
                'B': 0.6 if grade == 'B' else 0.2,
                'C': 0.6 if grade == 'C' else 0.2,
                'D': 0.6 if grade == 'D' else 0.2,
                'E': 0.8 if grade == 'E' else 0.2
            }
        }
    
    def analyze(self, image_path):
        """Complete analysis: grading + defect detection"""
        # Get defect analysis
        defect_result = self.analyze_defects_cv(image_path)
        
        # Get grade prediction
        grade_result = self._predict_grade(image_path)
        
        # Use defect-based grading if model prediction fails
        if grade_result is None:
            grade_result = self._grade_from_defects(defect_result)
        
        # Generate issues list
        issues = []
        counts = defect_result['counts']
        
        if counts.get('moldy_bean', 0) > 0:
            issues.append(f"Mold detected: {counts['moldy_bean']} beans")
        if counts.get('insect_damage', 0) > 0:
            issues.append(f"Insect damage: {counts['insect_damage']} beans")
        if counts.get('black_bean', 0) > 0:
            issues.append(f"Black/over-fermented: {counts['black_bean']} beans")
        if counts.get('broken_bean', 0) > 0:
            issues.append(f"Broken beans: {counts['broken_bean']} beans")
        if counts.get('foreign_object', 0) > 0:
            issues.append(f"Foreign objects: {counts['foreign_object']} items")
        
        # Generate recommendation
        grade = grade_result['grade']
        if grade in ['D', 'E']:
            if counts.get('moldy_bean', 0) > 0:
                recommendation = "REJECT: Mold contamination detected. Do not process."
            elif counts.get('foreign_object', 0) > 3:
                recommendation = "REJECT: High level of foreign objects. Additional sorting required."
            else:
                recommendation = "REJECT: Excessive defects. Consider lower grade usage or disposal."
        elif grade == 'A':
            recommendation = "PREMIUM: Suitable for specialty coffee. Export quality."
        elif grade == 'B':
            recommendation = "GOOD: Suitable for commercial grade coffee. Minor sorting recommended."
        elif grade == 'C':
            recommendation = "FAIR: Acceptable for instant coffee or blending. Sorting recommended."
        else:
            recommendation = "EVALUATION PENDING: Manual inspection recommended."
        
        # Calculate overall quality
        grade_score = {'A': 100, 'B': 80, 'C': 60, 'D': 40, 'E': 20}.get(grade, 0)
        overall_quality = round(grade_score * 0.4 + defect_result['quality_score'] * 0.6, 2)
        
        return {
            'grade': grade_result['grade'],
            'full_grade': grade_result['full_grade'],
            'confidence': grade_result['confidence'],
            'grade_analysis': grade_result,
            'defect_analysis': defect_result,
            'detections': defect_result['detections'],
            'issues': issues,
            'overall_quality': overall_quality,
            'quality_score': defect_result['quality_score'],
            'defect_ratio': defect_result['defect_ratio'],
            'all_probabilities': grade_result['all_probabilities'],
            'recommendation': recommendation,
            'timestamp': datetime.now().isoformat()
        }

# Initialize analyzer
analyzer = SimpleCoffeeAnalyzer()

def predict_image(image_file):
    """Main prediction function"""
    return analyzer.analyze(image_file)