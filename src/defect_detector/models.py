from django.db import models
from django.contrib.auth.models import User
import os

class DefectDataset(models.Model):
    """Tracks downloaded and custom datasets"""
    name = models.CharField(max_length=200)
    source = models.CharField(max_length=50, choices=[
        ('roboflow', 'Roboflow'),
        ('custom', 'Custom Upload'),
        ('combined', 'Combined'),
    ])
    roboflow_url = models.URLField(blank=True, null=True)
    local_path = models.CharField(max_length=500)
    num_images = models.IntegerField(default=0)
    num_classes = models.IntegerField(default=6)
    classes_list = models.JSONField(default=list)
    yaml_config = models.FileField(upload_to='dataset_configs/', blank=True)
    date_downloaded = models.DateTimeField(auto_now_add=True)
    is_active = models.BooleanField(default=True)
    
    def __str__(self):
        return f"{self.name} ({self.num_images} images)"

class TrainingSession(models.Model):
    """Tracks each training run"""
    dataset = models.ForeignKey(DefectDataset, on_delete=models.CASCADE)
    model_architecture = models.CharField(max_length=50, default='yolov8n')
    epochs = models.IntegerField(default=50)
    image_size = models.IntegerField(default=416)
    batch_size = models.IntegerField(default=16)
    custom_weight = models.FloatField(default=0.5, 
        help_text="0.0=public data only, 1.0=custom data only")
    
    status = models.CharField(max_length=20, choices=[
        ('pending', 'Pending'),
        ('running', 'Running'),
        ('completed', 'Completed'),
        ('failed', 'Failed'),
    ], default='pending')
    
    started_at = models.DateTimeField(auto_now_add=True)
    completed_at = models.DateTimeField(null=True, blank=True)
    
    # Results
    final_mAP50 = models.FloatField(null=True, blank=True)
    final_precision = models.FloatField(null=True, blank=True)
    final_recall = models.FloatField(null=True, blank=True)
    
    model_file = models.FileField(upload_to='trained_models/', blank=True)
    training_log = models.TextField(blank=True)
    
    def __str__(self):
        return f"Training {self.id} - {self.status}"

class DefectClass(models.Model):
    """Standardized defect class definitions"""
    name = models.CharField(max_length=50, unique=True)
    class_id = models.IntegerField()
    description = models.TextField()
    is_safety_critical = models.BooleanField(default=False)
    
    class Meta:
        ordering = ['class_id']
    
    def __str__(self):
        return f"{self.class_id}: {self.name}"