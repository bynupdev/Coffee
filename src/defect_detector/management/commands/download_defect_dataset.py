"""Management command to download defect dataset"""
import os
from django.core.management.base import BaseCommand
from defect_detector.utils.roboflow_downloader import RoboflowDownloader
from defect_detector.models import DefectDataset

class Command(BaseCommand):
    help = 'Download coffee bean defect dataset from Roboflow'
    
    def handle(self, *args, **options):
        self.stdout.write('='*50)
        self.stdout.write('Downloading Coffee Bean Defect Dataset')
        self.stdout.write('='*50)
        
        # Get API key
        api_key = os.environ.get('ROBOFLOW_API_KEY')
        
        if not api_key:
            self.stderr.write(self.style.ERROR(
                '\nROBOFLOW_API_KEY not found!\n'
                'Set it with: set ROBOFLOW_API_KEY=your_key_here\n'
                'Get your key from: https://app.roboflow.com (Settings > API)'
            ))
            return
        
        downloader = RoboflowDownloader(api_key=api_key)
        
        try:
            # Download the dataset
            # Using the Coffee Beans Defects dataset from Roboflow Universe
            save_dir = downloader.download_dataset(
                workspace='technserve-labs',
                project='coffee-beans-defects',
                version=1
            )
            
            # Count images
            train_dir = os.path.join(save_dir, 'train', 'images')
            val_dir = os.path.join(save_dir, 'valid', 'images')
            
            train_count = len([f for f in os.listdir(train_dir) 
                             if f.endswith(('.jpg', '.png', '.jpeg'))]) if os.path.exists(train_dir) else 0
            val_count = len([f for f in os.listdir(val_dir) 
                           if f.endswith(('.jpg', '.png', '.jpeg'))]) if os.path.exists(val_dir) else 0
            
            total_images = train_count + val_count
            
            # Create database record
            dataset = DefectDataset.objects.create(
                name='Coffee Beans Defects (Roboflow)',
                source='roboflow',
                local_path=save_dir,
                num_images=total_images,
                num_classes=7,
                classes_list=[
                    'good', 'dried_pod', 'foreign_matter',
                    'full_black', 'full_sour', 'fungus_damage',
                    'severe_insect_damage'
                ]
            )
            
            self.stdout.write(self.style.SUCCESS(
                f'\nDataset downloaded successfully!\n'
                f'Saved to: {save_dir}\n'
                f'Training images: {train_count}\n'
                f'Validation images: {val_count}\n'
                f'Total images: {total_images}\n'
                f'Database record ID: {dataset.id}\n'
            ))
            
        except Exception as e:
            self.stderr.write(self.style.ERROR(f'Download failed: {e}'))