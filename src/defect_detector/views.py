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
        # ===== STEP 2: BEAN MEASUREMENT (with full pixel debug) =====
        print("\n" + "-"*40)
        print("STEP 2: BEAN MEASUREMENT")
        print("-"*40)

        ORIGINAL_WIDTH = 800
        ORIGINAL_HEIGHT = 600
        QUALITY_MODEL_SIZE = 640

        if mm_per_pixel:
            try:
                project = workspace.project("coffee-bean-quality-jvz1r")
                version = project.version(1)
                quality_predictions = version.model.predict(temp_path, confidence=20).json()
                
                preds = quality_predictions.get('predictions', [])
                print(f"  Beans detected: {len(preds)}")
                
                # Store all pixel sizes for debugging
                all_pixel_sizes = []
                
                for i, pred in enumerate(preds):
                    cls = pred.get('class', '')
                    w_stretched = pred.get('width', 0)
                    h_stretched = pred.get('height', 0)
                    
                    # Store pixel dimensions from model
                    all_pixel_sizes.append({
                        'index': i,
                        'class': cls,
                        'w_640': w_stretched,
                        'h_640': h_stretched,
                        'area_640': w_stretched * h_stretched
                    })
                    
                    # Try BOTH correction directions to see which is right
                    # Option A: multiply (previous approach)
                    w_px_A = w_stretched * (ORIGINAL_WIDTH / QUALITY_MODEL_SIZE)
                    h_px_A = h_stretched * (ORIGINAL_HEIGHT / QUALITY_MODEL_SIZE)
                    
                    # Option B: divide (reverse approach)
                    w_px_B = w_stretched / (ORIGINAL_WIDTH / QUALITY_MODEL_SIZE)
                    h_px_B = h_stretched / (ORIGINAL_HEIGHT / QUALITY_MODEL_SIZE)
                    
                    # Use Option B (dividing) and store
                    w_px = w_px_B
                    h_px = h_px_B
                    
                    w_mm = round(w_px * mm_per_pixel, 1)
                    h_mm = round(h_px * mm_per_pixel, 1)
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
                
                # ===== DETAILED PIXEL DEBUG =====
                print(f"\n  === COIN vs BEAN PIXEL SIZES ===")
                print(f"  COIN: 80.5px in 640x640 = 22.0mm real")
                print(f"  Scale: {mm_per_pixel:.4f} mm/pixel")
                print(f"")
                print(f"  BEAN PIXEL SIZES (in 640x640 space):")
                print(f"  {'#':<4} {'Class':<15} {'W':<6} {'H':<6} {'Area':<8} {'W_mm':<8} {'H_mm':<8}")
                print(f"  {'-'*60}")
                
                for b in all_pixel_sizes[:15]:  # Show first 15 beans
                    bean_idx = b['index']
                    w_mm_val = bean_measurements[bean_idx]['width_mm'] if bean_idx < len(bean_measurements) else 0
                    h_mm_val = bean_measurements[bean_idx]['height_mm'] if bean_idx < len(bean_measurements) else 0
                    print(f"  {b['index']:<4} {b['class']:<15} {b['w_640']:<6.0f} {b['h_640']:<6.0f} {b['area_640']:<8.0f} {w_mm_val:<8.1f} {h_mm_val:<8.1f}")
                
                if len(all_pixel_sizes) > 15:
                    print(f"  ... and {len(all_pixel_sizes) - 15} more beans")
                
                # Calculate averages
                avg_w_px = sum(b['w_640'] for b in all_pixel_sizes) / len(all_pixel_sizes)
                avg_h_px = sum(b['h_640'] for b in all_pixel_sizes) / len(all_pixel_sizes)
                avg_area = sum(b['area_640'] for b in all_pixel_sizes) / len(all_pixel_sizes)
                
                print(f"\n  AVERAGES (in 640x640 space):")
                print(f"  Width: {avg_w_px:.1f}px | Height: {avg_h_px:.1f}px | Area: {avg_area:.0f}px²")
                print(f"  Coin comparison: Beans are {avg_w_px/80.5*100:.0f}% of coin width")
                print(f"  Expected: Beans should be ~25-35% of coin width (5-8mm vs 22mm)")
                
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
                    
            except Exception as e:
                print(f"  ❌ MEASUREMENT ERROR: {e}")
        else:
            print("  ⏭️ SKIPPED: No coin calibration available")

        # RIGHT AFTER the bean measurement loop, add this:
        print(f"\n  === PIXEL SIZE DEBUG ===")
        print(f"  COIN: 80.5px = 22.0mm")
        print(f"  SCALE: {mm_per_pixel:.4f} mm/pixel")
        print(f"  TOTAL BEANS: {len(bean_measurements)}")

        if bean_measurements:
            # Show first 5 beans
            for i in range(min(5, len(bean_measurements))):
                b = bean_measurements[i]
                print(f"  Bean {i}: w_px={b.get('width_px', 0):.0f}, h_px={b.get('height_px', 0):.0f}, d_mm={b['diameter_mm']}mm")
            
            # Show averages
            avg_w = sum(b.get('width_px', 0) for b in bean_measurements) / len(bean_measurements)
            avg_h = sum(b.get('height_px', 0) for b in bean_measurements) / len(bean_measurements)
            print(f"  AVG: w_px={avg_w:.0f}, h_px={avg_h:.0f}")
            print(f"  RATIO: beans are {avg_w/80.5*100:.0f}% of coin width")
            print(f"  EXPECTED: 25-35% (5-8mm beans vs 22mm coin)")

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
        
        # ===== STEP 4: SIZE-BASED GRADING =====
        print("\n" + "-"*40)
        print("STEP 4: SIZE-BASED GRADING")
        print("-"*40)
        
        if bean_measurements:
            diameters = [b['diameter_mm'] for b in bean_measurements]
            avg_size = sum(diameters) / len(diameters)
            
            # Industry standard screen sizes
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
            
            print(f"  Avg bean size: {avg_size:.1f}mm")
            print(f"  Size grade (before defects): {size_grade}")
            print(f"  Defect %: {defect_pct}%")
            
            # Defect penalty
            grades_order = ['AA', 'A', 'B', 'C', 'D']
            size_idx = grades_order.index(size_grade)
            
            if defect_pct > 30:
                # Heavy defects: downgrade 2 levels
                final_idx = min(size_idx + 2, 4)
                penalty = "DOWN 2"
            elif defect_pct > 15:
                # Moderate defects: downgrade 1 level
                final_idx = min(size_idx + 1, 4)
                penalty = "DOWN 1"
            else:
                # Few defects: keep size grade
                final_idx = size_idx
                penalty = "NONE"
            
            grade = grades_order[final_idx]
            
            print(f"  Penalty: {penalty}")
            print(f"  ✅ FINAL GRADE: {grade}")
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
        
        # ===== STEP 5: BEAN TYPE DETECTION =====
        # ===== STEP 5: BEAN TYPE DETECTION =====
        print("\n" + "-"*40)
        print("STEP 5: BEAN TYPE DETECTION")
        print("-"*40)

        # Constants for coordinate correction
        ORIGINAL_WIDTH = 800
        ORIGINAL_HEIGHT = 600
        QUALITY_MODEL_SIZE = 640

        # METHOD 1: Shape-based detection (from bean measurements)
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
                    print(f"  [SHAPE] Result: arabica (mostly oval)")
                elif oval_pct < 30:
                    shape_type = "robusta"
                    print(f"  [SHAPE] Result: robusta (mostly round)")
                else:
                    shape_type = "blend"
                    print(f"  [SHAPE] Result: blend (mixed shapes)")
        else:
            print(f"  [SHAPE] Not enough beans (need >5, have {len(bean_measurements) if bean_measurements else 0})")

        # METHOD 2: Public model detection
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
                conf = pred.get('confidence', 0)
                if cls not in type_counts:
                    type_counts[cls] = {'count': 0, 'total_conf': 0, 'max_conf': 0}
                type_counts[cls]['count'] += 1
                type_counts[cls]['total_conf'] += conf
                if conf > type_counts[cls]['max_conf']:
                    type_counts[cls]['max_conf'] = conf
            
            for cls, data in type_counts.items():
                avg_conf = (data['total_conf'] / data['count']) * 100 if data['count'] > 0 else 0
                max_conf = data['max_conf'] * 100
                print(f"  [MODEL] {cls}: {data['count']} preds, avg:{avg_conf:.0f}%, max:{max_conf:.0f}%")
            
            arabica_count = type_counts.get('arabica', {}).get('count', 0)
            robusta_count = type_counts.get('robusta', {}).get('count', 0)
            liberica_count = type_counts.get('liberica', {}).get('count', 0)
            
            print(f"  [MODEL] Raw - arabica:{arabica_count}, robusta:{robusta_count}, liberica:{liberica_count}")
            
            if arabica_count > robusta_count and arabica_count > liberica_count:
                model_type = "arabica"
            elif robusta_count > arabica_count and robusta_count > liberica_count:
                model_type = "robusta"
            elif liberica_count > arabica_count and liberica_count > robusta_count:
                model_type = "arabica"
            else:
                model_type = "arabica"
            
            print(f"  [MODEL] Result: {model_type}")
            
        except Exception as e:
            print(f"  [MODEL] Error: {e}")

        # METHOD 3: Size-based detection (with coordinate correction)
        size_type = "?"
        if bean_measurements and len(bean_measurements) > 5:
            diameters = []
            for b in bean_measurements:
                # Get raw pixel dimensions from quality model
                w_stretched = b.get('width_px', 0)
                h_stretched = b.get('height_px', 0)
                
                if w_stretched > 0 and h_stretched > 0:
                    # Convert from 640x640 back to 800x600
                    w_px = w_stretched * (ORIGINAL_WIDTH / QUALITY_MODEL_SIZE)
                    h_px = h_stretched * (ORIGINAL_HEIGHT / QUALITY_MODEL_SIZE)
                    
                    w_mm = w_px * mm_per_pixel
                    h_mm = h_px * mm_per_pixel
                    d_mm = (w_mm + h_mm) / 2
                    diameters.append(d_mm)
            
            if diameters:
                avg_diameter = sum(diameters) / len(diameters)
                min_diameter = min(diameters)
                max_diameter = max(diameters)
                
                print(f"  [SIZE] CORRECTED diameters: avg={avg_diameter:.1f}mm, min={min_diameter:.1f}mm, max={max_diameter:.1f}mm")
                print(f"  [SIZE] Reference: Arabica 5.5-8.0mm | Robusta 4.5-6.5mm")
                
                if avg_diameter < 5.8:
                    size_type = "robusta"
                    print(f"  [SIZE] Small beans (<5.8mm) → Robusta")
                elif avg_diameter > 6.8:
                    size_type = "arabica"
                    print(f"  [SIZE] Large beans (>6.8mm) → Arabica")
                else:
                    round_ratio = round_count / len(bean_measurements) if round_count > 0 else 0.5
                    if round_ratio > 0.6:
                        size_type = "robusta"
                        print(f"  [SIZE] Borderline size + mostly round → Robusta")
                    else:
                        size_type = "arabica"
                        print(f"  [SIZE] Borderline size + oval → Arabica")
        else:
            print(f"  [SIZE] Not enough beans for size analysis")

        # ===== FINAL DECISION =====
        print(f"\n  [FINAL] Shape: {shape_type} | Model: {model_type} | Size: {size_type}")

        votes_arabica = 0
        votes_robusta = 0

        if shape_type == "arabica": votes_arabica += 1
        elif shape_type == "robusta": votes_robusta += 1

        if model_type == "arabica": votes_arabica += 1
        elif model_type == "robusta": votes_robusta += 1

        if size_type == "arabica": votes_arabica += 1
        elif size_type == "robusta": votes_robusta += 1

        print(f"  [VOTE] Arabica: {votes_arabica}, Robusta: {votes_robusta}")

        if votes_arabica >= 2:
            bean_type = "arabica"
            print(f"  [VOTE] Majority Arabica")
        elif votes_robusta >= 2:
            bean_type = "robusta"
            print(f"  [VOTE] Majority Robusta")
        else:
            # Tie - trust shape (most reliable for your setup)
            if shape_type == "robusta":
                bean_type = "robusta"
            elif oval_count > 0 or round_count > 0:
                oval_pct_final = (oval_count * 100) // (oval_count + round_count)
                bean_type = "robusta" if oval_pct_final < 50 else "arabica"
            else:
                bean_type = "arabica"
            print(f"  [VOTE] Tie - using shape: {bean_type}")

        print(f"  ✅ FINAL TYPE: {bean_type}")
        
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
