from django.urls import path
from . import views

urlpatterns = [
    path('', views.index, name='defect_index'),
    # path('api/detect/', views.detect_defects_api, name='defect_detect'),
    # path('api/datasets/', views.dataset_list_api, name='defect_datasets'),
    # path('api/training-status/', views.training_status_api, name='defect_training_status'),
    path('api/analyze/', views.esp32_analysis_api, name='esp32_analyze'),
    path('api/status/', views.device_status_api, name='device_status'),
    path('api/datasets/', views.dataset_list_api, name='defect_datasets'),
    path('gallery/', views.view_captured_images, name='gallery'),
    path('image/<path:path>', views.serve_captured_image, name='serve_image'),
    
]