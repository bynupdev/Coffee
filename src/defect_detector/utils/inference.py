"""
Handles model inference using Roboflow Hosted API
"""
from roboflow import Roboflow
from collections import Counter
from django.conf import settings
import tempfile
import os

class ModelManager:
    """Manages all three detection models via Roboflow Hosted API"""
    
    def __init__(self):
        self.api_key = settings.ROBOFLOW_API_KEY
        self.rf = None
        self.models = {}
        self._initialized = False
        
        self.model_configs = {
            'foreign_matter': {
                'workspace': 'mfechos-coffee-workspace',
                'project': 'coffee-beans-defects-5hfat',
                'version': 1
            },
            'quality': {
                'workspace': 'mfechos-coffee-workspace',
                'project': 'coffee-bean-quality',
                'version': 1
            },
            'bean_type': {
                'workspace': 'mfechos-coffee-workspace',
                'project': 'coffee-bean-type-8i4hd',
                'version': 1
            }
        }
        
        self._ensure_initialized()
    
    def _ensure_initialized(self):
        """Initialize Roboflow connection"""
        if self._initialized:
            return
        
        print("Connecting to Roboflow...")
        self.rf = Roboflow(api_key=self.api_key)
        
        for model_name, config in self.model_configs.items():
            try:
                print(f"Loading {model_name} model...")
                workspace = self.rf.workspace(config['workspace'])
                project = workspace.project(config['project'])
                version = project.version(config['version'])
                self.models[model_name] = {
                    'workspace': workspace,
                    'project': project,
                    'version': version
                }
                print(f"  ✅ {model_name} model ready")
            except Exception as e:
                print(f"  ⚠️ Failed to load {model_name}: {e}")
                self.models[model_name] = None
        
        self._initialized = True
        print("✅ Roboflow connection established\n")
    
    def run_inference(self, image_bytes, enabled_detections):
        """Run selected models on the image"""
        self._ensure_initialized()
        
        from django.conf import settings
        import os as os_module
        
        media_dir = getattr(settings, 'MEDIA_ROOT', tempfile.gettempdir())
        os_module.makedirs(media_dir, exist_ok=True)
        temp_path = os_module.path.join(media_dir, 'coffee_analysis_temp.jpg')
        
        with open(temp_path, 'wb') as f:
            f.write(image_bytes)
        
        results = {}
        
        for detection_type in enabled_detections:
            model_data = self.models.get(detection_type)
            
            if model_data is None:
                results[detection_type] = {
                    'error': f'Model not available for {detection_type}'
                }
                continue
            
            try:
                # Use version.model.predict() - the model IS an ObjectDetectionModel
                version = model_data['version']
                predictions = version.model.predict(temp_path, confidence=40).json()
                
                if detection_type == 'foreign_matter':
                    results[detection_type] = self._parse_foreign_matter(predictions)
                elif detection_type == 'quality':
                    results[detection_type] = self._parse_classification(predictions)
                elif detection_type == 'bean_type':
                    results[detection_type] = self._parse_classification(predictions)
                        
            except Exception as e:
                import traceback
                print(f"Error in {detection_type}:")
                traceback.print_exc()
                results[detection_type] = {'error': str(e)[:80]}
        
        try:
            if os_module.path.exists(temp_path):
                os_module.remove(temp_path)
        except:
            pass
        
        return results

    def _parse_foreign_matter(self, predictions):
        """Parse foreign matter detection results"""
        detections = predictions.get('predictions', [])
        class_counts = Counter(d.get('class', 'unknown') for d in detections)
        
        return {
            'has_foreign_matter': 'foreign_matter' in class_counts,
            'foreign_count': class_counts.get('foreign_matter', 0),
            'total_detections': len(detections),
            'details': dict(class_counts)
        }
    
    def _parse_classification(self, predictions):
        """Parse classification results"""
        detections = predictions.get('predictions', [])
        
        if detections:
            top = max(detections, key=lambda x: x.get('confidence', 0))
            return {
                'prediction': top.get('class', 'unknown'),
                'confidence': round(top.get('confidence', 0), 3)
            }
        
        return {'prediction': 'unknown', 'confidence': 0}
    
    def get_available_models(self):
        """Return available models"""
        self._ensure_initialized()
        return {name: model is not None for name, model in self.models.items()}

# Global instance
model_manager = ModelManager()