"""
Coffee Bean Grading Model Training Script
This script will:
1. Download a coffee bean dataset
2. Train MobileNetV2 model
3. Save the model as coffee_grader.h5
"""

import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import os
import urllib.request
import zipfile
import shutil
from pathlib import Path

class CoffeeDatasetDownloader:
    """Download and prepare coffee bean dataset"""
    
    def __init__(self):
        self.base_dir = Path('coffee_dataset')
        self.train_dir = self.base_dir / 'train'
        self.val_dir = self.base_dir / 'validation'
        
    def download_coffee_dataset(self):
        """Download coffee bean dataset from public source"""
        print("📥 Downloading coffee bean dataset...")
        
        # Create directories
        self.base_dir.mkdir(exist_ok=True)
        self.train_dir.mkdir(exist_ok=True)
        self.val_dir.mkdir(exist_ok=True)
        
        # Download dataset (using a public coffee dataset URL)
        dataset_url = "https://storage.googleapis.com/coffee-bean-dataset/coffee_beans.zip"
        
        try:
            zip_path = self.base_dir / "coffee_beans.zip"
            urllib.request.urlretrieve(dataset_url, zip_path)
            
            # Extract dataset
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(self.base_dir)
            
            zip_path.unlink()  # Remove zip file
            print("✅ Dataset downloaded successfully")
            return True
            
        except Exception as e:
            print(f"⚠️ Could not download dataset: {e}")
            print("Creating synthetic dataset for demonstration...")
            return self.create_synthetic_dataset()
    
    def create_synthetic_dataset(self):
        """Create synthetic dataset for demonstration"""
        import cv2
        
        grades = ['Grade_A', 'Grade_B', 'Grade_C', 'Grade_D']
        
        for grade in grades:
            (self.train_dir / grade).mkdir(parents=True, exist_ok=True)
            (self.val_dir / grade).mkdir(parents=True, exist_ok=True)
        
        # Generate synthetic coffee bean images
        for grade_idx, grade in enumerate(grades):
            # Training images
            for i in range(50):
                img = self.generate_coffee_bean_image(grade_idx)
                img_path = self.train_dir / grade / f"coffee_{i:04d}.jpg"
                cv2.imwrite(str(img_path), img)
            
            # Validation images
            for i in range(10):
                img = self.generate_coffee_bean_image(grade_idx)
                img_path = self.val_dir / grade / f"coffee_{i:04d}.jpg"
                cv2.imwrite(str(img_path), img)
        
        print("✅ Synthetic dataset created")
        return True
    
    def generate_coffee_bean_image(self, grade):
        """Generate synthetic coffee bean image based on grade"""
        import cv2
        
        # Base image parameters
        img = np.zeros((300, 300, 3), dtype=np.uint8)
        
        # Grade-specific characteristics
        if grade == 0:  # Grade A - Perfect beans
            color = (70, 50, 30)  # Rich brown
            defects = 0
            size = (50, 70)
        elif grade == 1:  # Grade B - Good beans
            color = (80, 60, 40)
            defects = 2
            size = (45, 65)
        elif grade == 2:  # Grade C - Fair beans
            color = (90, 70, 50)
            defects = 5
            size = (40, 60)
        else:  # Grade D - Poor beans
            color = (100, 80, 60)
            defects = 10
            size = (35, 55)
        
        # Draw coffee bean shape
        center = (150, 150)
        axes = size
        cv2.ellipse(img, center, axes, 0, 0, 360, color, -1)
        
        # Add center line
        cv2.line(img, (center[0], center[1]-size[1]//2), 
                 (center[0], center[1]+size[1]//2), (40, 30, 20), 2)
        
        # Add defects (spots)
        for _ in range(defects):
            spot_x = np.random.randint(center[0]-size[0]//2, center[0]+size[0]//2)
            spot_y = np.random.randint(center[1]-size[1]//2, center[1]+size[1]//2)
            cv2.circle(img, (spot_x, spot_y), 3, (0, 0, 0), -1)
        
        return img

class CoffeeGraderTrainer:
    """Train MobileNetV2 model for coffee bean grading"""
    
    def __init__(self, img_size=224):
        self.img_size = img_size
        self.classes = ['Grade_A', 'Grade_B', 'Grade_C', 'Grade_D']
        self.model = None
        self.dataset_downloader = CoffeeDatasetDownloader()
        
    def prepare_data(self):
        """Prepare data generators"""
        print("📊 Preparing data generators...")
        
        # Download/prepare dataset
        self.dataset_downloader.download_coffee_dataset()
        
        # Data augmentation for training
        train_datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            fill_mode='nearest'
        )
        
        # Only rescaling for validation
        val_datagen = ImageDataGenerator(rescale=1./255)
        
        # Create generators
        self.train_generator = train_datagen.flow_from_directory(
            self.dataset_downloader.train_dir,
            target_size=(self.img_size, self.img_size),
            batch_size=32,
            class_mode='categorical',
            classes=self.classes
        )
        
        self.val_generator = val_datagen.flow_from_directory(
            self.dataset_downloader.val_dir,
            target_size=(self.img_size, self.img_size),
            batch_size=32,
            class_mode='categorical',
            classes=self.classes
        )
        
        print(f"✅ Found {self.train_generator.samples} training images")
        print(f"✅ Found {self.val_generator.samples} validation images")
        
    def build_model(self):
        """Build and compile the model"""
        print("🏗️ Building MobileNetV2 model...")
        
        # Load pre-trained MobileNetV2
        base_model = MobileNetV2(
            weights='imagenet',
            include_top=False,
            input_shape=(self.img_size, self.img_size, 3)
        )
        
        # Freeze base model layers
        base_model.trainable = False
        
        # Add custom classification layers
        self.model = models.Sequential([
            base_model,
            layers.GlobalAveragePooling2D(),
            layers.Dropout(0.3),
            layers.Dense(256, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            layers.Dense(128, activation='relu'),
            layers.BatchNormalization(),
            layers.Dense(4, activation='softmax')
        ])
        
        # Compile model
        self.model.compile(
            optimizer=optimizers.Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        print("✅ Model built successfully")
        self.model.summary()
        
    def train(self, epochs=20):
        """Train the model"""
        print(f"🚀 Starting training for {epochs} epochs...")
        
        # Callbacks
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor='val_accuracy',
                patience=5,
                restore_best_weights=True
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.2,
                patience=3,
                min_lr=0.00001
            ),
            tf.keras.callbacks.ModelCheckpoint(
                'best_model.h5',
                monitor='val_accuracy',
                save_best_only=True
            )
        ]
        
        # Train the model
        history = self.model.fit(
            self.train_generator,
            steps_per_epoch=self.train_generator.samples // 32,
            epochs=epochs,
            validation_data=self.val_generator,
            validation_steps=self.val_generator.samples // 32,
            callbacks=callbacks,
            verbose=1
        )
        
        print("✅ Initial training complete")
        
        # Fine-tuning
        print("🔧 Starting fine-tuning...")
        self.model.trainable = True
        
        # Recompile with lower learning rate
        self.model.compile(
            optimizer=optimizers.Adam(learning_rate=0.0001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        # Fine-tune
        history_fine = self.model.fit(
            self.train_generator,
            steps_per_epoch=self.train_generator.samples // 32,
            epochs=10,
            validation_data=self.val_generator,
            validation_steps=self.val_generator.samples // 32,
            verbose=1
        )
        
        print("✅ Fine-tuning complete")
        return history, history_fine
    
    def evaluate(self):
        """Evaluate the model"""
        print("📈 Evaluating model...")
        loss, accuracy = self.model.evaluate(self.val_generator)
        print(f"Validation Loss: {loss:.4f}")
        print(f"Validation Accuracy: {accuracy:.4f}")
        
    def save_model(self, filepath='coffee_grader.h5'):
        """Save the trained model"""
        self.model.save(filepath)
        print(f"✅ Model saved to {filepath}")

def main():
    """Main training function"""
    print("=" * 50)
    print("☕ Coffee Bean Grading System - Model Training")
    print("=" * 50)
    
    # Initialize trainer
    trainer = CoffeeGraderTrainer()
    
    # Prepare data
    trainer.prepare_data()
    
    # Build model
    trainer.build_model()
    
    # Train model
    trainer.train(epochs=20)
    
    # Evaluate model
    trainer.evaluate()
    
    # Save model to grader directory
    model_path = Path(__file__).parent / 'coffee_grader.h5'
    trainer.save_model(str(model_path))
    
    print("\n" + "=" * 50)
    print("✅ Training Complete!")
    print(f"Model saved to: {model_path}")
    print("You can now run the Django server: python manage.py runserver")
    print("=" * 50)

if __name__ == "__main__":
    main()