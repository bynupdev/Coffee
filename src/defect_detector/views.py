from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods
from .models import DefectDataset, TrainingSession

def index(request):
    """Main defect detection dashboard"""
    datasets = DefectDataset.objects.filter(is_active=True)
    training_sessions = TrainingSession.objects.all().order_by('-started_at')[:10]
    
    context = {
        'datasets': datasets,
        'training_sessions': training_sessions,
    }
    return render(request, 'defect_detector/index.html', context)

@csrf_exempt
@require_http_methods(["POST"])
def detect_defects_api(request):
    """API endpoint for defect detection"""
    # This will be implemented after training
    return JsonResponse({
        'status': 'ready',
        'message': 'Defect detection API is ready. Train a model first.'
    })

def dataset_list_api(request):
    """List all available datasets"""
    datasets = DefectDataset.objects.values(
        'id', 'name', 'source', 'num_images', 'num_classes'
    )
    return JsonResponse(list(datasets), safe=False)

def training_status_api(request):
    """Check training status"""
    sessions = TrainingSession.objects.all().order_by('-started_at')[:5]
    data = [{
        'id': s.id,
        'status': s.status,
        'started_at': s.started_at,
        'mAP50': s.final_mAP50,
    } for s in sessions]
    return JsonResponse(data, safe=False)



"""Views for defect detection app"""
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods
from django.conf import settings
import os
import tempfile
import base64

def get_model_manager():
    from .utils.inference import model_manager
    return model_manager

# @csrf_exempt
# @require_http_methods(["POST"])
# def esp32_analysis_api(request):
#     """
#     Main endpoint for ESP32-CAM devices
#     Accepts both raw binary and multipart form data
#     """
#     try:
#         model_manager = get_model_manager()
        
#         image_bytes = None
        
#         # Check if sent as form data (multipart)
#         if request.FILES and 'image' in request.FILES:
#             image_file = request.FILES['image']
#             image_bytes = image_file.read()
        
#         # Check if sent as raw body (ESP32 sends it this way)
#         elif request.body:
#             image_bytes = request.body
        
#         if not image_bytes:
#             return JsonResponse({
#                 'error': 'No image data found',
#                 'lines': ['ERROR', 'No image sent', '================'],
#                 'led_state': 'red'
#             }, status=400)
        
#         # Validate image size
#         if len(image_bytes) < 100:
#             return JsonResponse({
#                 'error': 'Image too small',
#                 'lines': ['ERROR', 'Invalid image', '================'],
#                 'led_state': 'red'
#             }, status=400)
        
#         if len(image_bytes) > 5 * 1024 * 1024:
#             return JsonResponse({
#                 'error': 'Image too large',
#                 'lines': ['ERROR', 'Image > 5MB', '================'],
#                 'led_state': 'red'
#             }, status=400)
        
#         # Get enabled detection types from header or query param
#         detection_header = request.headers.get('X-Detection-Types', 'foreign_matter,quality,bean_type')
#         enabled_detections = [d.strip() for d in detection_header.split(',') if d.strip()]
        
#         # Validate against available models
#         valid_types = list(model_manager.model_configs.keys())
#         enabled_detections = [d for d in enabled_detections if d in valid_types]
        
#         if not enabled_detections:
#             return JsonResponse({
#                 'error': 'No valid detection types',
#                 'lines': ['ERROR', 'No types enabled', '================'],
#                 'led_state': 'red'
#             }, status=400)
        
#         # Run inference
#         results = model_manager.run_inference(image_bytes, enabled_detections)
        
#         # Format for ESP32
#         esp32_response = format_for_esp32(results, enabled_detections)
        
#         return JsonResponse(esp32_response)
        
#     except Exception as e:
#         import traceback
#         traceback.print_exc()
#         return JsonResponse({
#             'error': str(e)[:100],
#             'lines': ['ERROR', str(e)[:30], '================'],
#             'led_state': 'red'
#         }, status=500)


@csrf_exempt
@require_http_methods(["POST"])
def esp32_analysis_api(request):
    """Main endpoint for ESP32-CAM devices - calls Roboflow API"""
    import os
    import traceback
    from datetime import datetime
    from django.conf import settings
    
    try:
        image_bytes = None
        
        if request.FILES and 'image' in request.FILES:
            image_file = request.FILES['image']
            image_bytes = image_file.read()
        elif request.body and len(request.body) > 100:
            image_bytes = request.body
        else:
            return JsonResponse({
                'error': 'No image data',
                'lines': ['ERROR', 'No image sent', '================'],
                'led_state': 'red'
            }, status=400)
        
        # Save image
        save_image = request.headers.get('X-Save-Image', 'false').lower() == 'true'
        if save_image:
            try:
                today = datetime.now()
                save_dir = os.path.join(settings.MEDIA_ROOT, 'captured_images',
                                       today.strftime('%Y'), today.strftime('%m'))
                os.makedirs(save_dir, exist_ok=True)
                filename = f"debug_{today.strftime('%Y%m%d_%H%M%S_%f')}.jpg"
                filepath = os.path.join(save_dir, filename)
                with open(filepath, 'wb') as f:
                    f.write(image_bytes)
                print(f"File saved! Size: {len(image_bytes)} bytes")
            except Exception as e:
                print(f"Save error: {e}")
        
        # Get detection types
        detection_header = request.headers.get('X-Detection-Types', 'foreign_matter,quality,bean_type')
        enabled = [d.strip() for d in detection_header.split(',') if d.strip()]
        
        # Call Roboflow for each model
        from roboflow import Roboflow
        rf = Roboflow(api_key=settings.ROBOFLOW_API_KEY)
        
        # Save temp file for Roboflow
        temp_path = os.path.join(settings.MEDIA_ROOT, 'temp_analysis.jpg')
        with open(temp_path, 'wb') as f:
            f.write(image_bytes)
        
        grade = "unknown"
        foreign = "None"
        bean_type = "unknown"
        
        # Foreign Matter Detection
        if 'foreign_matter' in enabled:
            try:
                workspace = rf.workspace("mfechos-coffee-workspace")
                project = workspace.project("coffee-beans-defects-5hfat")
                version = project.version(1)
                predictions = version.model.predict(temp_path, confidence=30).json()
                
                # Check for foreign matter in predictions
                for pred in predictions.get('predictions', []):
                    if pred.get('class') == 'foreign_matter':
                        foreign = "FOUND"
                        break
                    elif pred.get('class') in ['full_black', 'full_sour', 'fungus_damage', 'severe_insect_damage']:
                        foreign = pred.get('class', 'FOUND')
                        break
            except Exception as e:
                print(f"Foreign detection error: {e}")
        
        # Quality Grading
        if 'quality' in enabled:
            try:
                workspace = rf.workspace("mfechos-coffee-workspace")
                project = workspace.project("coffee-bean-quality")
                version = project.version(1)
                predictions = version.model.predict(temp_path, confidence=30).json()
                
                for pred in predictions.get('predictions', []):
                    predicted_class = pred.get('class', '')
                    if 'Grade' in predicted_class or predicted_class in ['A', 'B', 'C']:
                        grade = predicted_class.replace('Grade ', '')
                        break
            except Exception as e:
                print(f"Quality detection error: {e}")
        
        # Bean Type Detection
        if 'bean_type' in enabled:
            try:
                workspace = rf.workspace("mfechos-coffee-workspace")
                project = workspace.project("coffee-bean-type-8i4hd")
                version = project.version(1)
                predictions = version.model.predict(temp_path, confidence=30).json()
                
                for pred in predictions.get('predictions', []):
                    bean_type = pred.get('class', 'unknown')
                    break
            except Exception as e:
                print(f"Type detection error: {e}")
        
        # Clean up temp file
        if os.path.exists(temp_path):
            os.remove(temp_path)
        
        # Build response
        lines = [
            f"Foreign: {foreign}",
            f"Grade: Grade {grade}",
            f"Type: {bean_type}",
            "================"
        ]
        
        led_state = "green"
        if foreign != "None":
            led_state = "red"
        elif grade in ['C', 'D', 'E']:
            led_state = "yellow"
        
        return JsonResponse({
            'lines': lines,
            'led_state': led_state,
            'display_text': '\n'.join(lines)
        })
        
    except Exception as e:
        print(f"API error: {traceback.format_exc()}")
        return JsonResponse({
            'error': str(e)[:100],
            'lines': ['ERROR', str(e)[:30], '================'],
            'led_state': 'red'
        }, status=500)

@csrf_exempt
def device_status_api(request):
    """Health check and available models"""
    try:
        model_manager = get_model_manager()
        return JsonResponse({
            'status': 'ready',
            'models_available': model_manager.get_available_models(),
            'supported_detections': list(model_manager.model_configs.keys())
        })
    except Exception as e:
        return JsonResponse({
            'status': 'error',
            'error': str(e)
        }, status=500)


def format_for_esp32(results, enabled_detections):
    """Convert model results into ESP32-friendly format"""
    lines = []
    led_state = 'green'
    
    # Check for errors
    has_error = False
    for key, value in results.items():
        if isinstance(value, dict) and 'error' in value:
            lines.append(f"ERR:{value['error'][:12]}")
            has_error = True
    
    if has_error:
        return {
            'lines': lines if lines else ['Unknown error'],
            'led_state': 'red',
            'display_text': '\n'.join(lines) if lines else 'Error'
        }
    
    # Foreign Matter
    if 'foreign_matter' in results:
        fm = results['foreign_matter']
        if fm.get('has_foreign_matter'):
            count = fm.get('foreign_count', 0)
            lines.append(f"FOREIGN:{count} found!")
            led_state = 'red'
        else:
            lines.append("Foreign: None")
    
    # Quality
    if 'quality' in results:
        quality = results['quality'].get('prediction', '?')
        lines.append(f"Grade: {quality}")
        
        if quality in ['Grade C', 'Grade D', 'Grade E', 'C', 'D', 'E']:
            if led_state != 'red':
                led_state = 'yellow'
    
    # Bean Type
    if 'bean_type' in results:
        bean_type = results['bean_type'].get('prediction', '?')
        lines.append(f"Type: {bean_type}")
    
    if not lines:
        lines = ['No results']
    
    lines.append("================")
    
    return {
        'lines': lines,
        'led_state': led_state,
        'display_text': '\n'.join(lines)
    }



import os
from django.conf import settings
from django.http import HttpResponse, Http404
from django.shortcuts import render

def view_captured_images(request):
    """View to browse captured images"""
    images = []
    capture_dir = os.path.join(settings.MEDIA_ROOT, 'captured_images')
    
    if os.path.exists(capture_dir):
        for root, dirs, files in os.walk(capture_dir):
            for file in files:
                if file.endswith('.jpg'):
                    full_path = os.path.join(root, file)
                    rel_path = os.path.relpath(full_path, settings.MEDIA_ROOT)
                    images.append({
                        'name': file,
                        'url': f'/defects/image/{rel_path.replace(os.sep, "/")}'
                    })
    
    images.sort(key=lambda x: x['name'], reverse=True)
    return render(request, 'defect_detector/gallery.html', {
        'images': images[:20],
        'total': len(images)
    })

def serve_captured_image(request, path):
    """Serve captured images directly"""
    import re
    # Sanitize path to prevent directory traversal
    safe_path = re.sub(r'[^a-zA-Z0-9_/\.-]', '', path)
    full_path = os.path.join(settings.MEDIA_ROOT, safe_path)
    
    if os.path.exists(full_path) and full_path.endswith('.jpg'):
        with open(full_path, 'rb') as f:
            return HttpResponse(f.read(), content_type='image/jpeg')
    raise Http404("Image not found")
