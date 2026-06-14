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


import os
import traceback
from datetime import datetime
from django.conf import settings
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods
from roboflow import Roboflow

@csrf_exempt
@require_http_methods(["POST"])
def esp32_analysis_api(request):
    """Main endpoint - sends raw image directly to all three models"""
    import os
    import traceback
    from datetime import datetime
    from django.conf import settings
    from roboflow import Roboflow
    
    try:
        # Get image
        image_bytes = None
        if request.FILES and 'image' in request.FILES:
            image_bytes = request.FILES['image'].read()
        elif request.body and len(request.body) > 100:
            image_bytes = request.body
        else:
            return JsonResponse({'error': 'No image'}, status=400)
        
        # Save image
        today = datetime.now()
        save_dir = os.path.join(settings.MEDIA_ROOT, 'captured_images',
                               today.strftime('%Y'), today.strftime('%m'))
        os.makedirs(save_dir, exist_ok=True)
        temp_path = os.path.join(save_dir, f"capture_{today.strftime('%Y%m%d_%H%M%S')}.jpg")
        with open(temp_path, 'wb') as f:
            f.write(image_bytes)
        
        print(f"Image saved: {temp_path} ({len(image_bytes)} bytes)")
        
        # Connect to Roboflow
        rf = Roboflow(api_key=settings.ROBOFLOW_API_KEY)
        workspace = rf.workspace("mfechos-coffee-workspace")
        
        grade = "?"
        foreign = "None"
        bean_type = "?"
        
        # ===== QUALITY GRADING =====
        try:
            project = workspace.project("coffee-bean-quality")
            version = project.version(1)
            predictions = version.model.predict(temp_path, confidence=20).json()
            
            for pred in predictions.get('predictions', []):
                cls = pred.get('class', '')
                if 'A' in cls: grade = 'A'
                elif 'B' in cls: grade = 'B'
                elif 'C' in cls: grade = 'C'
                elif 'D' in cls: grade = 'D'
                break
            print(f"Grade: {grade}")
        except Exception as e:
            print(f"Quality error: {e}")
        
        # ===== FOREIGN MATTER (LOW THRESHOLD - catch everything) =====
        try:
            project = workspace.project("coffee-beans-defects-5hfat")
            version = project.version(1)
            predictions = version.model.predict(temp_path, confidence=5).json()
            
            # Count ALL classes
            all_counts = {}
            for pred in predictions.get('predictions', []):
                cls = pred.get('class', '')
                if cls not in all_counts:
                    all_counts[cls] = 0
                all_counts[cls] += 1
            
            print(f"All detections: {all_counts}")
            
            # Filter to defects only (exclude 'good')
            defect_classes = ['foreign_matter', 'full_black', 'full_sour', 
                            'fungus_damage', 'severe_insect_damage', 'dried_pod']
            
            defect_counts = {}
            for cls in defect_classes:
                if cls in all_counts:
                    defect_counts[cls] = all_counts[cls]
            
            good_count = all_counts.get('good', 0)
            total_defects = sum(defect_counts.values())
            
            if defect_counts:
                items = []
                for cls, count in defect_counts.items():
                    items.append(f"{cls}:{count}")
                foreign = ", ".join(items[:4])
                total = good_count + total_defects
                if total > 0:
                    pct = (total_defects * 100) // total
                    foreign += f"({pct}%)"
            else:
                foreign = "None"
            
            print(f"Foreign: {foreign}")
        except Exception as e:
            print(f"Foreign error: {traceback.format_exc()}")
        
        # ===== BEAN TYPE =====
        try:
            project = workspace.project("coffee-bean-type-8i4hd")
            version = project.version(1)
            predictions = version.model.predict(temp_path, confidence=20).json()
            
            for pred in predictions.get('predictions', []):
                bean_type = pred.get('class', '?')
                break
            print(f"Type: {bean_type}")
        except Exception as e:
            print(f"Type error: {e}")
        
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
