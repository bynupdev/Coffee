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
    """Main endpoint - uses coin calibration for accurate bean sizing"""
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
        
        # Constants
        COIN_DIAMETER_MM = 22.0  # Your coin: 2.2cm = 22mm
        
        # Results
        grade = "?"
        foreign = "None"
        bean_type = "?"
        size_grade = "?"
        mm_per_pixel = None
        bean_measurements = []
        total_beans_in_sample = 0
        defect_pct = 0
        
        # ===== STEP 1: FIND COIN FOR CALIBRATION =====
        try:
            project = workspace.project("coffee-defects-coin2")
            version = project.version(1)
            coin_predictions = version.model.predict(temp_path, confidence=50).json()
            
            for pred in coin_predictions.get('predictions', []):
                if pred.get('class', '').lower() == 'coin':
                    coin_width_px = pred.get('width', 0)
                    coin_height_px = pred.get('height', 0)
                    coin_diameter_px = (coin_width_px + coin_height_px) / 2
                    
                    mm_per_pixel = COIN_DIAMETER_MM / coin_diameter_px
                    
                    print(f"Coin found: {coin_diameter_px:.1f}px = {COIN_DIAMETER_MM}mm")
                    print(f"Calibration: {mm_per_pixel:.4f} mm/pixel")
                    break
            
            if mm_per_pixel is None:
                print("WARNING: No coin detected in image!")
                
        except Exception as e:
            print(f"Coin detection error: {e}")
        
        # ===== STEP 2: MEASURE BEANS (from quality model) =====
        if mm_per_pixel:
            try:
                project = workspace.project("coffee-bean-quality")
                version = project.version(1)
                quality_predictions = version.model.predict(temp_path, confidence=20).json()
                
                for pred in quality_predictions.get('predictions', []):
                    cls = pred.get('class', '')
                    width_px = pred.get('width', 0)
                    height_px = pred.get('height', 0)
                    
                    width_mm = round(width_px * mm_per_pixel, 1)
                    height_mm = round(height_px * mm_per_pixel, 1)
                    diameter_mm = round((width_mm + height_mm) / 2, 1)
                    
                    bean_measurements.append({
                        'class': cls,
                        'width_mm': width_mm,
                        'height_mm': height_mm,
                        'diameter_mm': diameter_mm
                    })
                
                total_beans_in_sample = len(bean_measurements)
                print(f"Measured {total_beans_in_sample} beans")
                
            except Exception as e:
                print(f"Bean measurement error: {e}")
        
        # ===== STEP 3: SIZE-BASED TYPE DETECTION =====
        if bean_measurements:
            oval_count = 0
            round_count = 0
            diameters = [b['diameter_mm'] for b in bean_measurements]
            
            for b in bean_measurements:
                w = b['width_mm']
                h = b['height_mm']
                if w > 0 and h > 0:
                    aspect_ratio = max(w, h) / min(w, h)
                    if aspect_ratio > 1.25:
                        oval_count += 1
                    else:
                        round_count += 1
            
            total_typed = oval_count + round_count
            if total_typed > 5:
                if oval_count > round_count * 2:
                    bean_type = "arabica"
                elif round_count > oval_count * 2:
                    bean_type = "robusta"
                else:
                    bean_type = "blend"
            
            # ===== STEP 4: SIZE-BASED GRADING =====
            if diameters:
                avg_size = sum(diameters) / len(diameters)
                min_size = min(diameters)
                max_size = max(diameters)
                
                variance = sum((d - avg_size) ** 2 for d in diameters) / len(diameters)
                std_dev = variance ** 0.5
                uniformity = round((1.0 - (std_dev / avg_size)) * 100) if avg_size > 0 else 0
                
                if avg_size >= 7.0:
                    size_grade = 'AA'
                elif avg_size >= 6.3:
                    size_grade = 'A'
                elif avg_size >= 5.5:
                    size_grade = 'B'
                elif avg_size >= 4.7:
                    size_grade = 'C'
                else:
                    size_grade = 'D'
                
                print(f"Sizes: avg={avg_size:.1f}mm, range={min_size:.1f}-{max_size:.1f}mm")
                print(f"Uniformity: {uniformity}%")
                print(f"Size grade: {size_grade}")
                print(f"Shape: oval={oval_count}, round={round_count}")
        
        # ===== STEP 5: DEFECT DETECTION (your custom model) =====
        try:
            project = workspace.project("my-coffee-defects")
            version = project.version(2)
            defect_predictions = version.model.predict(temp_path, confidence=1).json()
            
            all_counts = {}
            for pred in defect_predictions.get('predictions', []):
                cls = pred.get('class', '').lower()
                if cls == 'background':
                    continue
                if cls not in all_counts:
                    all_counts[cls] = 0
                all_counts[cls] += 1
            
            print(f"Defects found: {all_counts}")
            
            critical_defects = [
                'foreign_matter', 'foreign matter',
                'full_black', 'full black',
                'fungus_damage', 'fungus damage',
                'severe_insect_damage', 'severe insect damage',
                'broken'
            ]
            
            minor_defects = [
                'parchment', 'shell',
                'slight_insect_damage', 'slight insect damage',
                'immature'
            ]
            
            critical_counts = {}
            total_critical = 0
            for cls in critical_defects:
                count = all_counts.get(cls, 0)
                if count > 0:
                    simple = cls.lower().replace(' ', '_')[:20]
                    critical_counts[simple] = count
                    total_critical += count
            
            minor_counts = {}
            total_minor = 0
            for cls in minor_defects:
                count = all_counts.get(cls, 0)
                if count > 0:
                    simple = cls.lower().replace(' ', '_')[:20]
                    minor_counts[simple] = count
                    total_minor += count
            
            # Calculate defect percentage from TOTAL BEANS, not just defects
            if total_beans_in_sample > 0:
                defect_pct = min(100, (total_critical * 100) // total_beans_in_sample)
            elif sum(all_counts.values()) > 0:
                defect_pct = min(100, (total_critical * 100) // sum(all_counts.values()))
            else:
                defect_pct = 0
            
            # Build foreign matter result
            if critical_counts or minor_counts:
                parts = []
                for cls, count in critical_counts.items():
                    parts.append(f"{cls}:{count}")
                if minor_counts:
                    parts.append(f"minor:{total_minor}")
                foreign = ", ".join(parts[:4])
                foreign += f"({defect_pct}%)"
            else:
                foreign = "None"
                defect_pct = 0
            
            print(f"Critical: {total_critical}, Beans: {total_beans_in_sample}, Defect%: {defect_pct}%")
            
            # ===== STEP 6: COMBINED FINAL GRADE =====
            if total_beans_in_sample > 0:
                # Defect-based grade
                if defect_pct <= 5:
                    defect_grade = 'A'
                elif defect_pct <= 15:
                    defect_grade = 'B'
                elif defect_pct <= 30:
                    defect_grade = 'C'
                else:
                    defect_grade = 'D'
                
                # Combine with size grade if available
                if size_grade and size_grade != '?':
                    grades_order = ['AA', 'A', 'B', 'C', 'D']
                    defect_idx = grades_order.index(defect_grade) if defect_grade in grades_order else 1
                    size_idx = grades_order.index(size_grade) if size_grade in grades_order else 1
                    final_idx = max(defect_idx, size_idx)
                    grade = grades_order[final_idx]
                    print(f"Defect grade: {defect_grade}, Size grade: {size_grade}, Final: {grade}")
                else:
                    grade = defect_grade
            else:
                grade = 'A'
            
        except Exception as e:
            print(f"Defect detection error: {traceback.format_exc()}")
            foreign = "Error"
        
        # ===== FALLBACK TYPE DETECTION =====
        if bean_type == '?':
            try:
                project = workspace.project("coffee-bean-type-8i4hd")
                version = project.version(1)
                type_predictions = version.model.predict(temp_path, confidence=20).json()
                
                type_counts = {}
                for pred in type_predictions.get('predictions', []):
                    cls = pred.get('class', '').lower()
                    type_counts[cls] = type_counts.get(cls, 0) + 1
                
                arabica_count = type_counts.get('arabica', 0) + type_counts.get('liberica', 0)
                robusta_count = type_counts.get('robusta', 0)
                
                bean_type = "arabica" if arabica_count >= robusta_count else "robusta"
                print(f"Model-based type: {bean_type}")
            except:
                bean_type = "arabica"
        
        # ===== BUILD RESPONSE =====
        lines = [
            f"Foreign: {foreign}",
            f"Grade: Grade {grade}",
            f"Type: {bean_type}",
            "================"
        ]
        
        led_state = "green"
        if foreign != "None":
            led_state = "red"
        elif grade in ['C', 'D']:
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
