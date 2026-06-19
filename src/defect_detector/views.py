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
    """Main endpoint - size-based grading with coin calibration"""
    import os
    import traceback
    from datetime import datetime
    from django.conf import settings
    from roboflow import Roboflow
    
    try:
        print("\n" + "="*60)
        print("NEW REQUEST RECEIVED")
        print("="*60)
        
        # Get image
        image_bytes = None
        if request.FILES and 'image' in request.FILES:
            image_bytes = request.FILES['image'].read()
            print(f"[IMAGE] From FILES: {len(image_bytes)} bytes")
        elif request.body and len(request.body) > 100:
            image_bytes = request.body
            print(f"[IMAGE] From BODY: {len(image_bytes)} bytes")
        else:
            print("[ERROR] No image data found")
            return JsonResponse({'error': 'No image'}, status=400)
        
        # Save image
        today = datetime.now()
        save_dir = os.path.join(settings.MEDIA_ROOT, 'captured_images',
                               today.strftime('%Y'), today.strftime('%m'))
        os.makedirs(save_dir, exist_ok=True)
        temp_path = os.path.join(save_dir, f"capture_{today.strftime('%Y%m%d_%H%M%S')}.jpg")
        with open(temp_path, 'wb') as f:
            f.write(image_bytes)
        print(f"[SAVE] {temp_path}")
        
        # Connect to Roboflow
        print("[ROBOFLOW] Connecting...")
        rf = Roboflow(api_key=settings.ROBOFLOW_API_KEY)
        workspace = rf.workspace("mfechos-coffee-workspace")
        print("[ROBOFLOW] Connected to workspace")
        
        COIN_DIAMETER_MM = 22.0
        grade = "?"
        foreign = "None"
        bean_type = "?"
        mm_per_pixel = None
        bean_measurements = []
        total_beans_in_sample = 0
        defect_pct = 0
        size_grade = "?"
        avg_size = 0
        uniformity = 0
        
        # ===== STEP 1: COIN DETECTION & CALIBRATION =====
        print("\n" + "-"*40)
        print("STEP 1: COIN DETECTION")
        print("-"*40)
        try:
            project = workspace.project("coffee-defects-coin2")
            version = project.version(2)
            coin_predictions = version.model.predict(temp_path, confidence=50).json()
            
            preds = coin_predictions.get('predictions', [])
            print(f"  Predictions received: {len(preds)}")
            
            for pred in preds:
                cls = pred.get('class', '')
                conf = pred.get('confidence', 0)
                print(f"  Found: '{cls}' ({conf:.0%})")
                
                if cls.lower() == 'coin':
                    coin_w = pred.get('width', 0)
                    coin_h = pred.get('height', 0)
                    coin_diameter_px = (coin_w + coin_h) / 2
                    mm_per_pixel = COIN_DIAMETER_MM / coin_diameter_px
                    print(f"  ✅ COIN: {coin_diameter_px:.1f}px = {COIN_DIAMETER_MM}mm")
                    print(f"  ✅ CALIBRATION: {mm_per_pixel:.4f} mm/pixel")
                    break
            
            if mm_per_pixel is None:
                print("  ⚠️ WARNING: No coin detected!")
                
        except Exception as e:
            print(f"  ❌ COIN ERROR: {e}")
        
        # ===== STEP 2: BEAN MEASUREMENT =====
        # ===== STEP 2: BEAN MEASUREMENT (with real-world calibration) =====
        print("\n" + "-"*40)
        print("STEP 2: BEAN MEASUREMENT")
        print("-"*40)

        ORIGINAL_WIDTH = 800
        ORIGINAL_HEIGHT = 600
        QUALITY_MODEL_SIZE = 640

        # Calibration: Real Robusta beans = ~7mm diameter, System measures ~12mm
        BEAN_SIZE_CORRECTION = 0.58

        if mm_per_pixel:
            try:
                project = workspace.project("coffee-bean-quality-jvz1r")
                version = project.version(1)
                quality_predictions = version.model.predict(temp_path, confidence=20).json()
                
                preds = quality_predictions.get('predictions', [])
                print(f"  Beans detected: {len(preds)}")
                
                all_pixel_sizes = []
                
                for i, pred in enumerate(preds):
                    cls = pred.get('class', '')
                    w_stretched = pred.get('width', 0)
                    h_stretched = pred.get('height', 0)
                    
                    all_pixel_sizes.append({
                        'index': i,
                        'class': cls,
                        'w_640': w_stretched,
                        'h_640': h_stretched,
                        'area_640': w_stretched * h_stretched
                    })
                    
                    # Convert from 640x640 back to 800x600
                    w_px = w_stretched * (QUALITY_MODEL_SIZE / ORIGINAL_WIDTH)
                    h_px = h_stretched * (QUALITY_MODEL_SIZE / ORIGINAL_HEIGHT)
                    
                    # Convert to mm and apply correction
                    w_mm = round(w_px * mm_per_pixel * BEAN_SIZE_CORRECTION, 1)
                    h_mm = round(h_px * mm_per_pixel * BEAN_SIZE_CORRECTION, 1)
                    d_mm = round((w_mm + h_mm) / 2, 1)
                    
                    bean_measurements.append({
                        'class': cls,
                        'width_mm': w_mm,
                        'height_mm': h_mm,
                        'diameter_mm': d_mm,
                        'width_px': w_px,
                        'height_px': h_px
                    })
                
                total_beans_in_sample = len(bean_measurements)
                
                # Apply correction to mm_per_pixel for STEP 5
                mm_per_pixel = mm_per_pixel * BEAN_SIZE_CORRECTION
                
                # Debug output
                print(f"\n  === COIN vs BEAN PIXEL SIZES ===")
                print(f"  COIN: 80.5px in 640x640 = 22.0mm real")
                print(f"  Scale: {mm_per_pixel:.4f} mm/pixel (corrected)")
                print(f"")
                print(f"  BEAN SIZES (first 15):")
                print(f"  {'#':<4} {'Class':<15} {'W_px':<6} {'H_px':<6} {'W_mm':<8} {'H_mm':<8} {'Diam':<8}")
                print(f"  {'-'*65}")
                
                for b in all_pixel_sizes[:15]:
                    bean_idx = b['index']
                    if bean_idx < len(bean_measurements):
                        bm = bean_measurements[bean_idx]
                        print(f"  {b['index']:<4} {b['class']:<15} {b['w_640']:<6.0f} {b['h_640']:<6.0f} {bm['width_mm']:<8.1f} {bm['height_mm']:<8.1f} {bm['diameter_mm']:<8.1f}")
                
                if len(all_pixel_sizes) > 15:
                    print(f"  ... and {len(all_pixel_sizes) - 15} more beans")
                
                # Averages
                avg_w_px = sum(b['w_640'] for b in all_pixel_sizes) / len(all_pixel_sizes)
                avg_h_px = sum(b['h_640'] for b in all_pixel_sizes) / len(all_pixel_sizes)
                
                print(f"\n  AVERAGES (in 640x640 space):")
                print(f"  Width: {avg_w_px:.1f}px | Height: {avg_h_px:.1f}px")
                
                if bean_measurements:
                    diameters = [b['diameter_mm'] for b in bean_measurements]
                    avg_size = round(sum(diameters) / len(diameters), 1)
                    min_size = min(diameters)
                    max_size = max(diameters)
                    
                    variance = sum((d - avg_size) ** 2 for d in diameters) / len(diameters)
                    std_dev = variance ** 0.5
                    uniformity = round((1.0 - (std_dev / avg_size)) * 100) if avg_size > 0 else 0
                    
                    print(f"\n  📏 CORRECTED: Avg: {avg_size}mm | Range: {min_size}-{max_size}mm")
                    print(f"  📏 Uniformity: {uniformity}%")
                    print(f"  📏 Expected Robusta: 7mm avg | Arabica: 8mm avg")
                else:
                    print(f"  ⚠️ No beans measured")
                    
            except Exception as e:
                print(f"  ❌ MEASUREMENT ERROR: {e}")
        else:
            print("  ⏭️ SKIPPED: No coin calibration available")

        # ===== STEP 3: DEFECT DETECTION =====
        print("\n" + "-"*40)
        print("STEP 3: DEFECT DETECTION")
        print("-"*40)
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
            
            print(f"  All detections: {all_counts}")
            
            # Critical defects (affect grade)
            critical_defects = [
                'foreign_matter', 'foreign matter',
                'full_black', 'full black',
                'fungus_damage', 'fungus damage',
                'broken'
            ]
            
            # Minor defects (reported but don't affect grade)
            minor_defects = [
                'parchment', 'shell',
                'slight_insect_damage', 'slight insect damage',
                'severe_insect_damage', 'severe insect damage',
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
            
            print(f"  Critical: {critical_counts} (total: {total_critical})")
            print(f"  Minor: {minor_counts} (total: {total_minor})")
            
            # Calculate defect percentage
            if total_beans_in_sample > 0:
                defect_pct = min(100, (total_critical * 100) // total_beans_in_sample)
                print(f"  Defect %: {total_critical}/{total_beans_in_sample} = {defect_pct}%")
            else:
                defect_pct = 0
                print(f"  Defect %: N/A (no bean count)")
            
            # Build foreign result string
            if critical_counts or minor_counts:
                parts = []
                for cls, count in critical_counts.items():
                    parts.append(f"{cls}:{count}")
                if minor_counts:
                    parts.append(f"minor:{total_minor}")
                foreign = ", ".join(parts[:4])
                if total_beans_in_sample > 0:
                    foreign += f"({defect_pct}%)"
            else:
                foreign = "None"
            
            print(f"  Foreign result: {foreign}")
            
        except Exception as e:
            print(f"  ❌ DEFECT ERROR: {e}")

        

        # ===== STEP 4: BEAN TYPE DETECTION (with size-based logic) =====
        # ===== STEP 4: BEAN TYPE DETECTION (with clear 8.0/8.1 boundary) =====
        print("\n" + "-"*40)
        print("STEP 4: BEAN TYPE DETECTION")
        print("-"*40)

        # METHOD 1: Size-based detection (MOST RELIABLE)
        # Clear boundary: Robusta ≤ 8.0mm, Arabica ≥ 8.1mm
        size_type = "?"
        if bean_measurements and len(bean_measurements) > 5:
            diameters = []
            for b in bean_measurements:
                d_mm = b['diameter_mm']
                diameters.append(d_mm)
            
            if diameters:
                avg_diameter = sum(diameters) / len(diameters)
                min_diameter = min(diameters)
                max_diameter = max(diameters)
                
                # Count beans with clear boundary
                arabica_range = 0   # ≥ 8.1mm
                robusta_range = 0   # ≤ 8.0mm
                
                for d in diameters:
                    if d >= 8.1:
                        arabica_range += 1
                    else:
                        robusta_range += 1
                
                total = arabica_range + robusta_range
                
                print(f"  [SIZE] Bean diameters: avg={avg_diameter:.1f}mm, min={min_diameter:.1f}mm, max={max_diameter:.1f}mm")
                print(f"  [SIZE] Boundary: Robusta ≤ 8.0mm | Arabica ≥ 8.1mm")
                print(f"  [SIZE] Arabica (≥8.1mm): {arabica_range} beans ({arabica_range*100//total if total>0 else 0}%)")
                print(f"  [SIZE] Robusta (≤8.0mm): {robusta_range} beans ({robusta_range*100//total if total>0 else 0}%)")
                
                if arabica_range > robusta_range:
                    size_type = "arabica"
                    print(f"  [SIZE] More Arabica-range beans → Arabica")
                elif robusta_range > arabica_range:
                    size_type = "robusta"
                    print(f"  [SIZE] More Robusta-range beans → Robusta")
                else:
                    # Tie - use average diameter
                    size_type = "arabica" if avg_diameter >= 8.1 else "robusta"
                    print(f"  [SIZE] Tie - avg {avg_diameter:.1f}mm → {size_type}")
        else:
            print(f"  [SIZE] Not enough beans for size analysis (need >5, have {len(bean_measurements) if bean_measurements else 0})")

        # METHOD 2: Shape-based detection
        shape_type = "?"
        oval_count = 0
        round_count = 0

        if bean_measurements and len(bean_measurements) > 5:
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
            if total_typed > 0:
                oval_pct = (oval_count * 100) // total_typed
                print(f"  [SHAPE] Oval beans: {oval_count} ({oval_pct}%)")
                print(f"  [SHAPE] Round beans: {round_count} ({100 - oval_pct}%)")
                
                if oval_pct > 70:
                    shape_type = "arabica"
                elif oval_pct < 30:
                    shape_type = "robusta"
                else:
                    shape_type = "blend"
                print(f"  [SHAPE] Result: {shape_type}")

        # METHOD 3: Public model detection
        model_type = "?"
        try:
            project = workspace.project("coffee-bean-type-8i4hd")
            version = project.version(1)
            type_predictions = version.model.predict(temp_path, confidence=15).json()
            
            preds = type_predictions.get('predictions', [])
            print(f"  [MODEL] Total predictions: {len(preds)}")
            
            type_counts = {}
            for pred in preds:
                cls = pred.get('class', '').lower()
                if cls not in type_counts:
                    type_counts[cls] = 0
                type_counts[cls] += 1
            
            print(f"  [MODEL] Counts - {dict(type_counts)}")
            
            arabica_count = type_counts.get('arabica', 0) + type_counts.get('liberica', 0)
            robusta_count = type_counts.get('robusta', 0)
            
            model_type = "arabica" if arabica_count >= robusta_count else "robusta"
            print(f"  [MODEL] Result: {model_type}")
            
        except Exception as e:
            print(f"  [MODEL] Error: {e}")

        # ===== FINAL DECISION =====
        print(f"\n  [FINAL] Size: {size_type} | Shape: {shape_type} | Model: {model_type}")

        votes_arabica = 0
        votes_robusta = 0

        # Size gets 2 votes (most reliable - based on actual mm measurements)
        if size_type == "arabica": votes_arabica += 2
        elif size_type == "robusta": votes_robusta += 2

        # Shape gets 1 vote
        if shape_type == "arabica": votes_arabica += 1
        elif shape_type == "robusta": votes_robusta += 1

        # Model gets 1 vote
        if model_type == "arabica": votes_arabica += 1
        elif model_type == "robusta": votes_robusta += 1

        print(f"  [VOTE] Arabica: {votes_arabica}, Robusta: {votes_robusta} (Size=2, Shape=1, Model=1)")

        if votes_arabica > votes_robusta:
            bean_type = "arabica"
        elif votes_robusta > votes_arabica:
            bean_type = "robusta"
        else:
            # Tie - trust size (based on real measurements)
            bean_type = size_type if size_type != "?" else "arabica"
            print(f"  [VOTE] Tie - size wins: {bean_type}")

        print(f"  ✅ FINAL TYPE: {bean_type}")

        # Store for grading step
        detected_bean_type = bean_type


        # ===== STEP 5: SIZE-BASED GRADING (type-specific) =====
        # ===== STEP 5: SIZE-BASED GRADING (type-specific) =====
        print("\n" + "-"*40)
        print("STEP 5: SIZE-BASED GRADING")
        print("-"*40)

        if bean_measurements:
            diameters = [b['diameter_mm'] for b in bean_measurements]
            avg_size = sum(diameters) / len(diameters)
            min_size = min(diameters)
            max_size = max(diameters)
            
            variance = sum((d - avg_size) ** 2 for d in diameters) / len(diameters)
            std_dev = variance ** 0.5
            uniformity = round((1.0 - (std_dev / avg_size)) * 100) if avg_size > 0 else 0
            
            print(f"  Avg bean size: {avg_size:.1f}mm | Range: {min_size:.1f}-{max_size:.1f}mm")
            print(f"  Uniformity: {uniformity}%")
            print(f"  Bean type: {detected_bean_type}")
            
            # ===== TYPE-SPECIFIC SIZE GRADING =====
            if detected_bean_type == "arabica":
                # Arabica: 8-12mm expected, can achieve AA
                print(f"  Arabica standard: 8-12mm")
                if avg_size >= 10.0:
                    size_grade = 'AA'
                    print(f"    {avg_size:.1f}mm >= 10.0mm → AA (Premium)")
                elif avg_size >= 9.0:
                    size_grade = 'A'
                    print(f"    {avg_size:.1f}mm >= 9.0mm → A (Large)")
                elif avg_size >= 8.0:
                    size_grade = 'B'
                    print(f"    {avg_size:.1f}mm >= 8.0mm → B (Standard)")
                elif avg_size >= 6.5:
                    size_grade = 'C'
                    print(f"    {avg_size:.1f}mm >= 6.5mm → C (Small)")
                else:
                    size_grade = 'D'
                    print(f"    {avg_size:.1f}mm < 6.5mm → D (Undersized)")
            
            elif detected_bean_type == "robusta":
                # Robusta: 5-8mm expected, max grade is A (naturally smaller)
                print(f"  Robusta standard: 5-8mm")
                if avg_size >= 8.0:
                    size_grade = 'A'
                    print(f"    {avg_size:.1f}mm >= 8.0mm → A (Large Robusta)")
                elif avg_size >= 7.0:
                    size_grade = 'B'
                    print(f"    {avg_size:.1f}mm >= 7.0mm → B (Good)")
                elif avg_size >= 6.0:
                    size_grade = 'C'
                    print(f"    {avg_size:.1f}mm >= 6.0mm → C (Standard)")
                elif avg_size >= 5.0:
                    size_grade = 'D'
                    print(f"    {avg_size:.1f}mm >= 5.0mm → D (Small)")
                else:
                    size_grade = 'D'
                    print(f"    {avg_size:.1f}mm < 5.0mm → D (Reject)")
            
            else:
                # Generic fallback
                print(f"  Generic grading:")
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
                print(f"    {size_grade}")
            
            print(f"  Size grade (before defects): {size_grade}")
            print(f"  Defect %: {defect_pct}%")
            
            # ===== DEFECT PENALTY =====
            grades_order = ['AA', 'A', 'B', 'C', 'D']
            size_idx = grades_order.index(size_grade)
            
            if defect_pct > 30:
                final_idx = min(size_idx + 2, 4)
                penalty = "DOWN 2 (heavy defects)"
            elif defect_pct > 15:
                final_idx = min(size_idx + 1, 4)
                penalty = "DOWN 1 (moderate defects)"
            else:
                final_idx = size_idx
                penalty = "NONE (clean sample)"
            
            grade = grades_order[final_idx]
            
            print(f"  Penalty: {penalty}")
            print(f"  ✅ FINAL GRADE: {grade} ({detected_bean_type})")
            
        else:
            # Fallback to defect-only grading
            print(f"  ⚠️ No size data, using defect-only grading")
            if defect_pct <= 5:
                grade = 'A'
            elif defect_pct <= 15:
                grade = 'B'
            elif defect_pct <= 30:
                grade = 'C'
            else:
                grade = 'D'
            print(f"  ✅ FALLBACK GRADE: {grade}")

        # ===== BUILD FINAL RESPONSE =====
        print("\n" + "="*60)
        print("FINAL RESULT")
        print("="*60)
        print(f"  Foreign: {foreign}")
        print(f"  Grade: {grade} (size: {size_grade}, defect%: {defect_pct}%, uniformity: {uniformity}%)")
        print(f"  Type: {bean_type}")
        print(f"  Bean count: {total_beans_in_sample}, Avg size: {avg_size}mm")
        print("="*60 + "\n")
        
        lines = [
            f"Foreign: {foreign}",
            f"Grade: Grade {grade}",
            f"Type: {bean_type}",
            "================"
        ]
        
        led_state = "green"
        if foreign != "None" and defect_pct > 10:
            led_state = "red"
        elif grade in ['C', 'D']:
            led_state = "yellow"
        
        return JsonResponse({
            'lines': lines,
            'led_state': led_state,
            'display_text': '\n'.join(lines)
        })
        
    except Exception as e:
        print(f"\n❌ FATAL ERROR: {traceback.format_exc()}\n")
        return JsonResponse({
            'lines': ['ERROR', str(e)[:30], '================'],
            'led_state': 'red'
        }, status=500)




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
