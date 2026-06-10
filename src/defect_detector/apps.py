# from django.apps import AppConfig


# class DefectDetectorConfig(AppConfig):
#     name = 'defect_detector'


from django.apps import AppConfig

class DefectDetectorConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'defect_detector'
    verbose_name = 'Coffee Bean Defect Detection'
    
    def ready(self):
        # Import signals if needed later
        pass