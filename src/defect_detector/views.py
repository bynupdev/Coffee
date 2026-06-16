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

# @csrf_exempt
# @require_http_methods(["POST"])
# def esp32_analysis_api(request):
#     """Main endpoint - sends raw image directly to all three models"""
#     import os
#     import traceback
#     from datetime import datetime
#     from django.conf import settings
#     from roboflow import Roboflow
    
#     try:
#         # Get image
#         image_bytes = None
#         if request.FILES and 'image' in request.FILES:
#             image_bytes = request.FILES['image'].read()
#         elif request.body and len(request.body) > 100:
#             image_bytes = request.body
#         else:
#             return JsonResponse({'error': 'No image'}, status=400)
        
#         # Save image
#         today = datetime.now()
#         save_dir = os.path.join(settings.MEDIA_ROOT, 'captured_images',
#                                today.strftime('%Y'), today.strftime('%m'))
#         os.makedirs(save_dir, exist_ok=True)
#         temp_path = os.path.join(save_dir, f"capture_{today.strftime('%Y%m%d_%H%M%S')}.jpg")
#         with open(temp_path, 'wb') as f:
#             f.write(image_bytes)
        
#         print(f"Image saved: {temp_path} ({len(image_bytes)} bytes)")
        
#         # Connect to Roboflow
#         rf = Roboflow(api_key=settings.ROBOFLOW_API_KEY)
#         workspace = rf.workspace("mfechos-coffee-workspace")
        
#         grade = "?"
#         foreign = "None"
#         bean_type = "?"
        
#         # ===== QUALITY GRADING =====
#         try:
#             project = workspace.project("coffee-bean-quality")
#             version = project.version(1)
#             predictions = version.model.predict(temp_path, confidence=20).json()
            
#             for pred in predictions.get('predictions', []):
#                 cls = pred.get('class', '')
#                 if 'A' in cls: grade = 'A'
#                 elif 'B' in cls: grade = 'B'
#                 elif 'C' in cls: grade = 'C'
#                 elif 'D' in cls: grade = 'D'
#                 break
#             print(f"Grade: {grade}")
#         except Exception as e:
#             print(f"Quality error: {e}")
        
#         # ===== FOREIGN MATTER (LOW THRESHOLD - catch everything) =====
#         # ===== FOREIGN MATTER (Segmentation Model) - Case Insensitive =====
#         try:
#             project = workspace.project("coffee-beans-dataset-2-segmentation-peuoq")
#             version = project.version(1)
#             predictions = version.model.predict(temp_path, confidence=40).json()
            
#             # Critical defects (affect grade significantly)
#             critical_defects = [
#                 'foreign matter', 'full black', 'fungus damage', 
#                 'severe insect damage', 'broken'
#             ]
            
#             # Minor defects (common in raw coffee, less impact on grade)
#             minor_defects = [
#                 'parchment', 'shell', 'slight insect damage', 'immature'
#             ]
            
#             # Count everything (convert to lowercase)
#             all_counts = {}
#             for pred in predictions.get('predictions', []):
#                 cls = pred.get('class', '').lower()
#                 if cls not in all_counts:
#                     all_counts[cls] = 0
#                 all_counts[cls] += 1
            
#             print(f"All detections: {all_counts}")
            
#             good_count = all_counts.get('good', 0)
            
#             # Count critical defects
#             critical_counts = {}
#             total_critical = 0
#             for cls in critical_defects:
#                 count = all_counts.get(cls, 0)
#                 if count > 0:
#                     critical_counts[cls] = count
#                     total_critical += count
            
#             # Count minor defects
#             minor_counts = {}
#             total_minor = 0
#             for cls in minor_defects:
#                 count = all_counts.get(cls, 0)
#                 if count > 0:
#                     minor_counts[cls] = count
#                     total_minor += count
            
#             total_defects = total_critical + total_minor
#             total_objects = sum(all_counts.values())
            
#             # Build result string
#             if critical_counts or minor_counts:
#                 parts = []
                
#                 # Show critical defects first (most important)
#                 for cls, count in critical_counts.items():
#                     short = cls.replace(' ', '_')[:18]
#                     parts.append(f"{short}:{count}")
                
#                 # Show minor defect total if any
#                 if minor_counts:
#                     parts.append(f"minor:{total_minor}")
                
#                 foreign = ", ".join(parts[:4])
                
#                 # Calculate percentage based on critical defects
#                 if total_objects > 0:
#                     pct = (total_critical * 100) // total_objects
#                     foreign += f"({pct}%)"
#             else:
#                 foreign = "None"
            
#             print(f"Good: {good_count}, Critical: {total_critical}, Minor: {total_minor}, Total: {total_objects}")
#             print(f"Foreign result: {foreign}")
            
#         except Exception as e:
#             print(f"Foreign error: {traceback.format_exc()}")
#             foreign = "Error"

#         # ===== BEAN TYPE =====
        
#         # ===== BEAN TYPE (Arabica/Robusta only) =====
#         try:
#             project = workspace.project("coffee-bean-type-8i4hd")
#             version = project.version(1)
#             predictions = version.model.predict(temp_path, confidence=20).json()
            
#             type_counts = {}
#             for pred in predictions.get('predictions', []):
#                 cls = pred.get('class', '').lower()
#                 if cls not in type_counts:
#                     type_counts[cls] = 0
#                 type_counts[cls] += 1
            
#             print(f"Type counts: {type_counts}")
            
#             # Get the most common type
#             arabica_count = type_counts.get('arabica', 0)
#             robusta_count = type_counts.get('robusta', 0)
#             liberica_count = type_counts.get('liberica', 0)
            
#             # Map liberica to arabica (they look similar)
#             arabica_count += liberica_count
            
#             if arabica_count >= robusta_count:
#                 bean_type = "arabica"
#             else:
#                 bean_type = "robusta"
            
#             print(f"Type: {bean_type} (arabica:{arabica_count}, robusta:{robusta_count})")
            
#         except Exception as e:
#             print(f"Type error: {e}")
#             bean_type = "unrecognised"  # Default to arabica


#         # Build response
#         lines = [
#             f"Foreign: {foreign}",
#             f"Grade: Grade {grade}",
#             f"Type: {bean_type}",
#             "================"
#         ]
        
#         led_state = "green"
#         if foreign != "None":
#             led_state = "red"
#         elif grade in ['C', 'D', 'E']:
#             led_state = "yellow"
        
#         return JsonResponse({
#             'lines': lines,
#             'led_state': led_state,
#             'display_text': '\n'.join(lines)
#         })
        
#     except Exception as e:
#         print(f"API error: {traceback.format_exc()}")
#         return JsonResponse({
#             'lines': ['ERROR', str(e)[:30], '================'],
#             'led_state': 'red'
#         }, status=500)


@csrf_exempt
@require_http_methods(["POST"])
def esp32_analysis_api(request):
    """Main endpoint - uses custom trained model for defect detection"""
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
        bean_type = "arabica"
        
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
        
        # ===== FOREIGN MATTER (Your Custom Model - my-coffee-defects/2) =====
        
        # ===== FOREIGN MATTER (Your Custom Model) =====
        try:
            project = workspace.project("my-coffee-defects")
            version = project.version(2)
            predictions = version.model.predict(temp_path, confidence=0).json()
            
            print(f"=== FOREIGN MATTER DEBUG ===")
            print(f"Total predictions: {len(predictions.get('predictions', []))}")
            
            # Print EVERY detection with confidence
            all_counts = {}
            for pred in predictions.get('predictions', []):
                cls = pred.get('class', '')
                conf = pred.get('confidence', 0)
                
                # Convert to lowercase for matching
                cls_lower = cls.lower()
                
                print(f"  Class: {cls} (lower: {cls_lower}) | Confidence: {conf:.4f}")
                
                if cls_lower == 'background':
                    continue
                    
                if cls_lower not in all_counts:
                    all_counts[cls_lower] = 0
                all_counts[cls_lower] += 1
            
            print(f"All counts (non-background): {all_counts}")
            
            # Critical defects - try multiple name formats
            critical_defects = [
                'foreign_matter', 'foreign matter', 'foreign_matter', 'Foreign_matter',
                'full_black', 'full black', 'Full_black',
                'fungus_damage', 'fungus damage', 'Fungus_damage',
                'severe_insect_damage', 'severe insect damage', 'Severe_insect_damage',
                'broken', 'Broken'
            ]
            
            # Minor defects
            minor_defects = [
                'parchment', 'Parchment',
                'shell', 'Shell',
                'slight_insect_damage', 'slight insect damage', 'Slight_insect_damage',
                'immature', 'Immature'
            ]
            
            critical_counts = {}
            total_critical = 0
            for cls in critical_defects:
                count = all_counts.get(cls, 0)
                if count > 0:
                    # Use simplified name
                    simple_name = cls.lower().replace(' ', '_')[:20]
                    if simple_name not in critical_counts:
                        critical_counts[simple_name] = 0
                    critical_counts[simple_name] += count
                    total_critical += count
            
            minor_counts = {}
            total_minor = 0
            for cls in minor_defects:
                count = all_counts.get(cls, 0)
                if count > 0:
                    simple_name = cls.lower().replace(' ', '_')[:20]
                    if simple_name not in minor_counts:
                        minor_counts[simple_name] = 0
                    minor_counts[simple_name] += count
                    total_minor += count
            
            total_objects = sum(all_counts.values())
            
            print(f"Critical counts: {critical_counts} (total: {total_critical})")
            print(f"Minor counts: {minor_counts} (total: {total_minor})")
            print(f"Total objects: {total_objects}")
            
            # In the foreign matter section, replace the percentage calculation:

            if critical_counts or minor_counts:
                parts = []
                for cls, count in critical_counts.items():
                    parts.append(f"{cls}:{count}")
                
                if minor_counts:
                    parts.append(f"minor:{total_minor}")
                
                foreign = ", ".join(parts[:4])
                
                # Don't show percentage if we can't calculate it properly
                # Only show percentage if there are non-critical objects too
                if total_objects > total_critical and total_objects > 0:
                    pct = (total_critical * 100) // total_objects
                    foreign += f"({pct}%)"

            print(f"Foreign result: {foreign}")
            print(f"=== END DEBUG ===")
            
        except Exception as e:
            print(f"Foreign error: {traceback.format_exc()}")
            foreign = "Error"


        # ===== BEAN TYPE (Arabica/Robusta only - map liberica to arabica) =====
        try:
            project = workspace.project("coffee-bean-type-8i4hd")
            version = project.version(1)
            predictions = version.model.predict(temp_path, confidence=20).json()
            
            type_counts = {}
            for pred in predictions.get('predictions', []):
                cls = pred.get('class', '').lower()
                if cls not in type_counts:
                    type_counts[cls] = 0
                type_counts[cls] += 1
            
            arabica_count = type_counts.get('arabica', 0)
            robusta_count = type_counts.get('robusta', 0)
            liberica_count = type_counts.get('liberica', 0)
            
            # Map liberica to arabica (they look similar)
            arabica_count += liberica_count
            
            if arabica_count >= robusta_count:
                bean_type = "arabica"
            else:
                bean_type = "robusta"
            
            print(f"Type counts: {type_counts}")
            print(f"Type result: {bean_type}")
            
        except Exception as e:
            print(f"Type error: {e}")
            bean_type = "arabica"
        
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
