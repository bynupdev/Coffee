"""Enhanced views with training control and detailed analysis"""
from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods
from django.core.files.storage import default_storage
from django.core.files.base import ContentFile
from django.conf import settings
import os
import json
import uuid
from pathlib import Path
import threading
import shutil

from .advanced_trainer import AdvancedCoffeeGrader

# Global trainer instance
trainer = AdvancedCoffeeGrader(base_dir=Path(__file__).parent.parent)
training_status = {'is_training': False, 'progress': 0, 'message': ''}

def index(request):
    """Render main page with training controls"""
    return render(request, 'grader/index.html')

@csrf_exempt
@require_http_methods(["POST"])
def predict_api(request):
    """API endpoint for coffee grade prediction"""
    import traceback
    import sys
    
    try:
        # Check if image was uploaded
        if 'image' not in request.FILES:
            return JsonResponse({'error': 'No image provided'}, status=400)
        
        image_file = request.FILES['image']
        
        # Basic validation
        if not image_file.content_type.startswith('image/'):
            return JsonResponse({'error': 'File must be an image'}, status=400)
        
        # Read the image data
        image_data = image_file.read()
        
        # Try simple analysis first
        from .prediction import predict_image
        
        try:
            result = predict_image(image_data)
            return JsonResponse(result)
        except Exception as pred_error:
            # If prediction fails, return a basic analysis
            print(f"Prediction error details: {traceback.format_exc()}", file=sys.stderr)
            
            # Return a fallback response
            import numpy as np
            import cv2
            
            # Decode image for basic analysis
            nparr = np.frombuffer(image_data, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if img is not None:
                h, w = img.shape[:2]
                return JsonResponse({
                    'grade': 'C',
                    'confidence': 0.5,
                    'full_grade': 'Grade_C',
                    'quality_score': 50.0,
                    'defect_ratio': 0.15,
                    'issues': ['Analysis limited - model prediction failed'],
                    'overall_quality': 50.0,
                    'recommendation': 'Manual inspection recommended',
                    'error_details': str(pred_error)[:200],
                    'image_size': f"{w}x{h}"
                })
            else:
                return JsonResponse({
                    'error': f'Failed to process image: {str(pred_error)}',
                    'status': 'error'
                }, status=500)
            
    except Exception as e:
        print(f"View error: {traceback.format_exc()}", file=sys.stderr)
        return JsonResponse({
            'error': str(e),
            'status': 'error'
        }, status=500)
    
@csrf_exempt
@require_http_methods(["POST"])
def upload_custom_images(request):
    """Upload and organize custom training images"""
    try:
        grade = request.POST.get('grade')
        images = request.FILES.getlist('images')
        
        if not grade or grade not in trainer.classes:
            return JsonResponse({'error': 'Invalid grade'}, status=400)
        
        # Create grade directory
        grade_dir = trainer.custom_dataset_dir / grade
        grade_dir.mkdir(parents=True, exist_ok=True)
        
        saved_files = []
        for image in images:
            filename = f"{uuid.uuid4()}.jpg"
            filepath = grade_dir / filename
            with open(filepath, 'wb') as f:
                for chunk in image.chunks():
                    f.write(chunk)
            saved_files.append(filename)
        
        return JsonResponse({
            'success': True,
            'message': f'Uploaded {len(saved_files)} images to {grade}',
            'files': saved_files
        })
        
    except Exception as e:
        return JsonResponse({'error': str(e)}, status=500)

@csrf_exempt
@require_http_methods(["POST"])
def start_training(request):
    """Start model training with custom weight"""
    global training_status
    
    if training_status['is_training']:
        return JsonResponse({'error': 'Training already in progress'}, status=400)
    
    try:
        data = json.loads(request.body)
        custom_weight = float(data.get('custom_weight', 0.5))
        epochs = int(data.get('epochs', 30))
        
        # Start training in background thread
        def train_thread():
            global training_status
            training_status = {'is_training': True, 'progress': 0, 'message': 'Starting...'}
            
            try:
                trainer.train(custom_weight=custom_weight, epochs=epochs)
                training_status['message'] = 'Training complete!'
                training_status['progress'] = 100
            except Exception as e:
                training_status['message'] = f'Error: {str(e)}'
            finally:
                training_status['is_training'] = False
        
        thread = threading.Thread(target=train_thread)
        thread.start()
        
        return JsonResponse({
            'success': True,
            'message': 'Training started',
            'custom_weight': custom_weight
        })
        
    except Exception as e:
        return JsonResponse({'error': str(e)}, status=500)

@require_http_methods(["GET"])
def training_status_api(request):
    """Get current training status"""
    return JsonResponse(training_status)

@require_http_methods(["GET"])
def dataset_stats(request):
    """Get custom dataset statistics"""
    stats = {}
    total_images = 0
    
    for grade in trainer.classes:
        grade_dir = trainer.custom_dataset_dir / grade
        if grade_dir.exists():
            count = len(list(grade_dir.glob('*.jpg'))) + \
                   len(list(grade_dir.glob('*.jpeg'))) + \
                   len(list(grade_dir.glob('*.png')))
            stats[grade] = count
            total_images += count
        else:
            stats[grade] = 0
    
    return JsonResponse({
        'total_images': total_images,
        'by_grade': stats,
        'classes': trainer.classes
    })

@csrf_exempt
@require_http_methods(["POST"])
def analyze_sample(request):
    """Analyze a sample image for defects"""
    try:
        if 'image' not in request.FILES:
            return JsonResponse({'error': 'No image provided'}, status=400)
        
        image_file = request.FILES['image']
        temp_filename = f"sample_{uuid.uuid4()}.jpg"
        temp_path = default_storage.save(
            os.path.join('uploads', temp_filename),
            ContentFile(image_file.read())
        )
        
        full_path = os.path.join(settings.MEDIA_ROOT, temp_path)
        
        try:
            analysis = trainer.analyze_bean_defects(full_path)
            return JsonResponse(analysis)
        finally:
            if default_storage.exists(temp_path):
                default_storage.delete(temp_path)
                
    except Exception as e:
        return JsonResponse({'error': str(e)}, status=500)
    

# Add these imports at the top
from .defect_detector import CoffeeDefectDetector
from .prediction import analyzer, predict_image

# Add these new views
@csrf_exempt
@require_http_methods(["POST"])
def detect_defects_api(request):
    """API endpoint for defect detection with visualization"""
    try:
        if 'image' not in request.FILES:
            return JsonResponse({'error': 'No image provided'}, status=400)
        
        image_file = request.FILES['image']
        temp_filename = f"defect_{uuid.uuid4()}.jpg"
        temp_path = default_storage.save(
            os.path.join('uploads', temp_filename),
            ContentFile(image_file.read())
        )
        
        full_path = os.path.join(settings.MEDIA_ROOT, temp_path)
        
        try:
            # Run defect detection
            result = analyzer.analyze_single_image(full_path)
            
            # Generate visualized image
            vis_filename = f"visualized_{uuid.uuid4()}.jpg"
            vis_path = os.path.join(settings.MEDIA_ROOT, 'uploads', vis_filename)
            
            detector = CoffeeDefectDetector()
            detector.visualize_detections(full_path, vis_path)
            
            result['visualization_url'] = f"/media/uploads/{vis_filename}"
            
            return JsonResponse(result)
            
        finally:
            if default_storage.exists(temp_path):
                default_storage.delete(temp_path)
                
    except Exception as e:
        return JsonResponse({'error': str(e)}, status=500)

@csrf_exempt
@require_http_methods(["POST"])
def analyze_sample_api(request):
    """API endpoint for batch sample analysis"""
    try:
        images = request.FILES.getlist('images')
        
        if not images:
            return JsonResponse({'error': 'No images provided'}, status=400)
        
        # Save all images temporarily
        temp_paths = []
        for image in images:
            temp_filename = f"sample_{uuid.uuid4()}.jpg"
            temp_path = default_storage.save(
                os.path.join('uploads', temp_filename),
                ContentFile(image.read())
            )
            temp_paths.append(os.path.join(settings.MEDIA_ROOT, temp_path))
        
        try:
            # Analyze sample batch
            result = analyzer.analyze_sample_batch(temp_paths)
            
            sample_id = str(uuid.uuid4())
            result['sample_id'] = sample_id
            
            return JsonResponse(result)
            
        finally:
            # Clean up temp files
            for temp_path in temp_paths:
                if default_storage.exists(temp_path):
                    default_storage.delete(temp_path)
                
    except Exception as e:
        return JsonResponse({'error': str(e)}, status=500)

@csrf_exempt
@require_http_methods(["POST"])
def train_defect_model_api(request):
    """API endpoint to train defect detection model"""
    try:
        data = json.loads(request.body)
        dataset_dir = data.get('dataset_dir', 'defect_dataset')
        epochs = int(data.get('epochs', 15))
        
        # Start training in background
        def train_thread():
            try:
                analyzer.train_defect_detector(dataset_dir, epochs)
            except Exception as e:
                print(f"Training error: {e}")
        
        thread = threading.Thread(target=train_thread)
        thread.start()
        
        return JsonResponse({
            'success': True,
            'message': 'Defect detection model training started'
        })
        
    except Exception as e:
        return JsonResponse({'error': str(e)}, status=500)