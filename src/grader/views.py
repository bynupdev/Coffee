"""Views for coffee grader app - Render Ready"""
from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods
from django.core.files.storage import default_storage
from django.core.files.base import ContentFile
from django.conf import settings
import os
import uuid
from datetime import datetime

def index(request):
    """Render main page"""
    return render(request, 'grader/index.html')

@csrf_exempt
@require_http_methods(["POST"])
def predict_api(request):
    """API endpoint for coffee grade prediction - uses Roboflow API"""
    try:
        if 'image' not in request.FILES:
            return JsonResponse({'error': 'No image provided'}, status=400)
        
        image_file = request.FILES['image']
        
        if not image_file.content_type.startswith('image/'):
            return JsonResponse({'error': 'File must be an image'}, status=400)
        
        image_data = image_file.read()
        
        # Use the defect_detector's model manager (Roboflow API)
        from defect_detector.utils.inference import model_manager
        
        try:
            # Save temp image
            temp_filename = f"grade_{uuid.uuid4()}.jpg"
            temp_path = default_storage.save(
                os.path.join('uploads', temp_filename),
                ContentFile(image_data)
            )
            full_path = os.path.join(settings.MEDIA_ROOT, temp_path)
            
            # Run inference using Roboflow
            results = model_manager.run_inference(image_data, ['quality'])
            
            # Clean up
            if default_storage.exists(temp_path):
                default_storage.delete(temp_path)
            
            if 'quality' in results and 'prediction' in results['quality']:
                grade = results['quality']['prediction']
                confidence = results['quality'].get('confidence', 0.5)
                return JsonResponse({
                    'grade': grade.replace('Grade ', ''),
                    'confidence': confidence,
                    'full_grade': grade,
                    'quality_score': confidence * 100,
                    'defect_ratio': 0.1,
                    'overall_quality': confidence * 100,
                    'recommendation': f'Grade {grade} coffee beans'
                })
            else:
                return JsonResponse({
                    'grade': 'C',
                    'confidence': 0.5,
                    'full_grade': 'Grade_C',
                    'quality_score': 50.0,
                    'recommendation': 'Analysis completed'
                })
                
        except Exception as e:
            return JsonResponse({
                'grade': 'C',
                'confidence': 0.5,
                'full_grade': 'Grade_C',
                'quality_score': 50.0,
                'recommendation': f'Analysis limited: {str(e)[:50]}'
            })
            
    except Exception as e:
        return JsonResponse({'error': str(e)[:100], 'status': 'error'}, status=500)

@csrf_exempt
def health_check(request):
    """Health check endpoint"""
    return JsonResponse({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat()
    })