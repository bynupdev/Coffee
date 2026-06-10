from django.urls import path
from . import views

urlpatterns = [
    path('', views.index, name='index'),
    path('predict/', views.predict_api, name='predict'),
    path('upload-custom/', views.upload_custom_images, name='upload_custom'),
    path('start-training/', views.start_training, name='start_training'),
    path('training-status/', views.training_status_api, name='training_status'),
    path('dataset-stats/', views.dataset_stats, name='dataset_stats'),
    path('analyze-sample/', views.analyze_sample, name='analyze_sample'),
    path('detect-defects/', views.detect_defects_api, name='detect_defects'),
    path('analyze-sample/', views.analyze_sample_api, name='analyze_sample'),
    path('train-defect-model/', views.train_defect_model_api, name='train_defect_model'),
]