"""
Advanced Coffee Bean Grading System
Supports custom dataset with configurable weight control
"""

import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.applications import MobileNetV2, EfficientNetB0
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
from PIL import Image
import os
import json
from pathlib import Path
import shutil
from datetime import datetime
import cv2

class AdvancedCoffeeGrader:
    """Advanced coffee grading with defect analysis"""
    
    def __init__(self, base_dir='.'):
        self.base_dir = Path(base_dir)
        self.custom_dataset_dir = self.base_dir / 'custom_dataset'
        self.models_dir = self.base_dir / 'models'
        self.models_dir.mkdir(exist_ok=True)
        
        # Extended grade classes including extremely bad
        self.classes = ['Grade_A', 'Grade_B', 'Grade_C', 'Grade_D', 'Grade_E']
        self.grade_descriptions = {
            'Grade_A': 'Premium - No defects, uniform size, excellent color',
            'Grade_B': 'Good - Minor defects, good uniformity',
            'Grade_C': 'Fair - Some defects, acceptable quality',
            'Grade_D': 'Poor - Multiple defects, inconsistent',
            'Grade_E': 'Reject - Severe defects, mold, insect damage'
        }
        
        self.img_size = (224, 224)
        self.model = None
        self.defect_detector = None
        
    def organize_custom_images(self, source_dir=None):
        """
        Organize custom images into grade folders
        Supports drag-and-drop folder structure
        """
        print("📁 Organizing custom dataset...")
        
        # Create grade directories if they don't exist
        for grade in self.classes:
            (self.custom_dataset_dir / grade).mkdir(parents=True, exist_ok=True)
        
        if source_dir:
            source_path = Path(source_dir)
            if source_path.exists():
                # Copy images maintaining structure
                for grade in self.classes:
                    grade_source = source_path / grade
                    if grade_source.exists():
                        grade_target = self.custom_dataset_dir / grade
                        for img_file in grade_source.glob('*'):
                            if img_file.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                                shutil.copy2(img_file, grade_target / img_file.name)
        
        # Count images
        counts = {}
        for grade in self.classes:
            grade_dir = self.custom_dataset_dir / grade
            counts[grade] = len(list(grade_dir.glob('*.jpg'))) + \
                           len(list(grade_dir.glob('*.jpeg'))) + \
                           len(list(grade_dir.glob('*.png')))
        
        print("✅ Dataset organization complete:")
        for grade, count in counts.items():
            print(f"   {grade}: {count} images")
        
        return counts
    
    def analyze_bean_defects(self, image_path):
        """
        Analyze individual coffee bean for defects
        Returns defect percentage and quality metrics
        """
        img = cv2.imread(str(image_path))
        if img is None:
            return None
        
        # Convert to HSV for better color analysis
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        
        # Define color ranges for defects
        # Black/very dark beans (over-roasted or mold)
        lower_black = np.array([0, 0, 0])
        upper_black = np.array([180, 255, 50])
        black_mask = cv2.inRange(hsv, lower_black, upper_black)
        
        # White/pale beans (underdeveloped)
        lower_white = np.array([0, 0, 200])
        upper_white = np.array([180, 30, 255])
        white_mask = cv2.inRange(hsv, lower_white, upper_white)
        
        # Brown/yellow (good beans)
        lower_brown = np.array([10, 50, 50])
        upper_brown = np.array([30, 255, 200])
        brown_mask = cv2.inRange(hsv, lower_brown, upper_brown)
        
        # Calculate percentages
        total_pixels = img.shape[0] * img.shape[1]
        black_pixels = cv2.countNonZero(black_mask)
        white_pixels = cv2.countNonZero(white_mask)
        brown_pixels = cv2.countNonZero(brown_mask)
        
        # Defect calculations
        defect_percentage = ((black_pixels + white_pixels) / total_pixels) * 100
        quality_score = (brown_pixels / total_pixels) * 100
        
        # Detect bean count and size uniformity
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (11, 11), 0)
        _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter small contours (noise)
        bean_contours = [c for c in contours if cv2.contourArea(c) > 100]
        bean_count = len(bean_contours)
        
        # Calculate size uniformity
        if bean_count > 0:
            areas = [cv2.contourArea(c) for c in bean_contours]
            mean_area = np.mean(areas)
            std_area = np.std(areas)
            uniformity = 1 - (std_area / mean_area) if mean_area > 0 else 0
        else:
            uniformity = 0
        
        return {
            'defect_percentage': round(defect_percentage, 2),
            'quality_score': round(quality_score, 2),
            'bean_count': bean_count,
            'uniformity': round(uniformity * 100, 2),
            'black_defects': round((black_pixels / total_pixels) * 100, 2),
            'white_defects': round((white_pixels / total_pixels) * 100, 2)
        }
    
    def create_data_generators(self, custom_weight=0.5):
        """
        Create data generators with configurable custom dataset weight
        custom_weight: 0.0 = only base dataset, 1.0 = only custom dataset
        """
        print(f"📊 Creating data generators (Custom weight: {custom_weight * 100:.0f}%)")
        
        # Data augmentation
        train_datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=30,
            width_shift_range=0.3,
            height_shift_range=0.3,
            shear_range=0.2,
            zoom_range=0.3,
            horizontal_flip=True,
            vertical_flip=False,
            brightness_range=[0.8, 1.2],
            fill_mode='nearest',
            validation_split=0.2
        )
        
        val_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)
        
        # Load custom dataset
        custom_train_gen = None
        custom_val_gen = None
        
        if custom_weight > 0 and self.custom_dataset_dir.exists():
            custom_train_gen = train_datagen.flow_from_directory(
                self.custom_dataset_dir,
                target_size=self.img_size,
                batch_size=32,
                class_mode='categorical',
                classes=self.classes,
                subset='training'
            )
            
            custom_val_gen = val_datagen.flow_from_directory(
                self.custom_dataset_dir,
                target_size=self.img_size,
                batch_size=32,
                class_mode='categorical',
                classes=self.classes,
                subset='validation'
            )
        
        # For demonstration, create synthetic base dataset
        base_train_gen = self.create_synthetic_generator(train_datagen, 'training')
        base_val_gen = self.create_synthetic_generator(val_datagen, 'validation')
        
        # Combine generators based on weight
        if custom_weight == 0:
            return base_train_gen, base_val_gen
        elif custom_weight == 1:
            return custom_train_gen, custom_val_gen
        else:
            # Create combined generator
            combined_train_gen = CombinedGenerator(
                base_train_gen, custom_train_gen, custom_weight
            )
            combined_val_gen = CombinedGenerator(
                base_val_gen, custom_val_gen, custom_weight
            )
            return combined_train_gen, combined_val_gen
    
    def create_synthetic_generator(self, datagen, subset):
        """Create synthetic data for demonstration"""
        # Create temporary directory with synthetic images
        temp_dir = self.base_dir / 'temp_synthetic'
        temp_dir.mkdir(exist_ok=True)
        
        for grade in self.classes:
            (temp_dir / grade).mkdir(exist_ok=True)
            # Generate synthetic images
            for i in range(50 if subset == 'training' else 10):
                img = self.generate_synthetic_bean(grade)
                img.save(temp_dir / grade / f'synth_{i:04d}.jpg')
        
        generator = datagen.flow_from_directory(
            temp_dir,
            target_size=self.img_size,
            batch_size=32,
            class_mode='categorical',
            classes=self.classes,
            subset=subset
        )
        
        return generator
    
    def generate_synthetic_bean(self, grade):
        """Generate synthetic coffee bean image"""
        img = Image.new('RGB', (300, 300), color=(200, 180, 150))
        
        # Grade-specific characteristics
        grade_idx = self.classes.index(grade)
        
        if grade_idx == 0:  # Grade A
            color = (80, 50, 30)
            spots = 0
        elif grade_idx == 1:  # Grade B
            color = (90, 60, 40)
            spots = 2
        elif grade_idx == 2:  # Grade C
            color = (100, 70, 50)
            spots = 5
        elif grade_idx == 3:  # Grade D
            color = (110, 80, 60)
            spots = 10
        else:  # Grade E - Extremely bad
            color = (120, 90, 70)
            spots = 20
        
        # Convert to numpy for drawing
        img_array = np.array(img)
        
        # Draw coffee bean shape
        center = (150, 150)
        axes = (60 - grade_idx * 5, 80 - grade_idx * 5)
        cv2.ellipse(img_array, center, axes, 0, 0, 360, color, -1)
        
        # Add defects
        for _ in range(spots):
            x = np.random.randint(center[0] - axes[0]//2, center[0] + axes[0]//2)
            y = np.random.randint(center[1] - axes[1]//2, center[1] + axes[1]//2)
            cv2.circle(img_array, (x, y), 3, (0, 0, 0), -1)
        
        return Image.fromarray(img_array)
    
    def build_advanced_model(self):
        """Build advanced model with defect detection branch"""
        print("🏗️ Building advanced model...")
        
        # Base model for feature extraction
        base_model = EfficientNetB0(
            weights='imagenet',
            include_top=False,
            input_shape=(224, 224, 3)
        )
        base_model.trainable = False
        
        # Grade classification branch
        x = base_model.output
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dropout(0.3)(x)
        x = layers.Dense(512, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.5)(x)
        x = layers.Dense(256, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        
        # Grade output
        grade_output = layers.Dense(len(self.classes), activation='softmax', name='grade')(x)
        
        # Defect analysis branch
        defect_branch = layers.Dense(128, activation='relu')(x)
        defect_branch = layers.Dropout(0.3)(defect_branch)
        defect_output = layers.Dense(1, activation='sigmoid', name='defect_score')(defect_branch)
        
        # Quality score branch
        quality_branch = layers.Dense(128, activation='relu')(x)
        quality_branch = layers.Dropout(0.3)(quality_branch)
        quality_output = layers.Dense(1, activation='linear', name='quality_score')(quality_branch)
        
        # Create model
        self.model = models.Model(
            inputs=base_model.input,
            outputs=[grade_output, defect_output, quality_output]
        )
        
        # Compile with multiple outputs
        self.model.compile(
            optimizer=optimizers.Adam(learning_rate=0.001),
            loss={
                'grade': 'categorical_crossentropy',
                'defect_score': 'binary_crossentropy',
                'quality_score': 'mse'
            },
            loss_weights={
                'grade': 1.0,
                'defect_score': 0.5,
                'quality_score': 0.5
            },
            metrics={
                'grade': 'accuracy',
                'defect_score': 'mae',
                'quality_score': 'mae'
            }
        )
        
        print("✅ Advanced model built")
        return self.model
    
    def train(self, custom_weight=0.5, epochs=30):
        """Train the model with specified custom dataset weight"""
        print(f"\n🚀 Starting training with {custom_weight*100:.0f}% custom data")
        
        # Save training configuration
        config = {
            'custom_weight': custom_weight,
            'epochs': epochs,
            'classes': self.classes,
            'timestamp': datetime.now().isoformat()
        }
        
        with open(self.base_dir / 'training_config.json', 'w') as f:
            json.dump(config, f, indent=2)
        
        # Create generators
        train_gen, val_gen = self.create_data_generators(custom_weight)
        
        # Build model
        self.build_advanced_model()
        
        # Callbacks
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor='val_grade_accuracy',
                patience=7,
                restore_best_weights=True
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.2,
                patience=4,
                min_lr=0.00001
            ),
            tf.keras.callbacks.ModelCheckpoint(
                str(self.models_dir / 'best_model.h5'),
                monitor='val_grade_accuracy',
                save_best_only=True
            )
        ]
        
        # Train
        history = self.model.fit(
            train_gen,
            epochs=epochs,
            validation_data=val_gen,
            callbacks=callbacks,
            verbose=1
        )
        
        # Save final model
        model_path = self.models_dir / f'custom_model_weight_{custom_weight:.2f}.h5'
        self.model.save(str(model_path))
        
        print(f"\n✅ Training complete! Model saved to {model_path}")
        return history
    
    def predict_with_analysis(self, image_path):
        """
        Comprehensive prediction with grade, defects, and quality
        """
        if self.model is None:
            model_path = self.models_dir / 'best_model.h5'
            if model_path.exists():
                self.model = tf.keras.models.load_model(str(model_path))
            else:
                raise Exception("No trained model found")
        
        # Load and preprocess image
        img = Image.open(image_path)
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        # Analyze defects using computer vision
        cv_analysis = self.analyze_bean_defects(image_path)
        
        # Prepare for model prediction
        img_resized = img.resize(self.img_size)
        img_array = np.array(img_resized) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        
        # Model prediction
        predictions = self.model.predict(img_array, verbose=0)
        grade_probs = predictions[0][0]
        model_defect_score = predictions[1][0][0]
        model_quality_score = predictions[2][0][0]
        
        # Get predicted grade
        grade_idx = np.argmax(grade_probs)
        grade = self.classes[grade_idx].split('_')[1]
        full_grade = self.classes[grade_idx]
        confidence = float(grade_probs[grade_idx])
        
        # Combine CV and model analysis
        combined_defect_score = (
            cv_analysis['defect_percentage'] / 100 * 0.4 + 
            model_defect_score * 0.6
        ) if cv_analysis else model_defect_score
        
        combined_quality_score = (
            cv_analysis['quality_score'] * 0.4 + 
            model_quality_score * 0.6
        ) if cv_analysis else model_quality_score * 100
        
        return {
            'grade': grade,
            'full_grade': full_grade,
            'confidence': round(confidence, 4),
            'grade_description': self.grade_descriptions[full_grade],
            'defect_percentage': round(combined_defect_score * 100, 2),
            'quality_score': round(combined_quality_score, 2),
            'all_probabilities': {
                self.classes[i].split('_')[1]: round(float(prob), 4)
                for i, prob in enumerate(grade_probs)
            },
            'cv_analysis': cv_analysis,
            'model_scores': {
                'defect_score': round(float(model_defect_score), 4),
                'quality_score': round(float(model_quality_score), 4)
            }
        }

class CombinedGenerator:
    """Generator that combines base and custom datasets with weighting"""
    
    def __init__(self, base_gen, custom_gen, custom_weight):
        self.base_gen = base_gen
        self.custom_gen = custom_gen
        self.custom_weight = custom_weight
        self.base_weight = 1 - custom_weight
        
    def __iter__(self):
        return self
    
    def __next__(self):
        # Randomly choose between base and custom based on weights
        if np.random.random() < self.custom_weight:
            return next(self.custom_gen)
        else:
            return next(self.base_gen)