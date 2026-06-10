"""Management command to download defect dataset"""
import os
from django.core.management.base import BaseCommand
from django.conf import settings
from defect_detector.utils.roboflow_downloader import RoboflowDownloader
from defect_detector.models import DefectDataset

class Command(BaseCommand):
    help = 'Download coffee bean defect dataset from Roboflow'
    
    def add_arguments(self, parser):
        parser.add_argument(
            '--source',
            type=str,
            default='universe',
            choices=['universe', 'workspace'],
            help='Download from Roboflow Universe or your workspace'
        )
        parser.add_argument(
            '--workspace',
            type=str,
            help='Roboflow workspace name (your username for forked datasets)'
        )
    
    def handle(self, *args, **options):
        self.stdout.write('='*60)
        self.stdout.write('☕ Downloading Coffee Bean Defect Dataset')
        self.stdout.write('='*60)
        
        # Get API key
        api_key = os.environ.get('ROBOFLOW_API_KEY') or getattr(settings, 'ROBOFLOW_API_KEY', None)
        
        if not api_key:
            self.stderr.write(self.style.ERROR(
                '\n❌ ROBOFLOW_API_KEY not found!\n'
                'Get your key from: https://app.roboflow.com (Settings > API)\n'
                'Then run: set ROBOFLOW_API_KEY=your_key_here'
            ))
            return
        
        downloader = RoboflowDownloader(api_key=api_key)
        
        source = options['source']
        
        try:
            if source == 'universe':
                # Download from Roboflow Universe (public dataset)
                self.stdout.write('\n📥 Downloading from Roboflow Universe...')
                
                save_dir = downloader.download_dataset(
                    workspace='technserve-labs',
                    project='coffee-beans-defects',
                    version=1
                )
            else:
                # Download from your workspace (forked dataset)
                workspace = options['workspace']
                if not workspace:
                    self.stderr.write(self.style.ERROR(
                        'Please provide --workspace with your Roboflow username'
                    ))
                    return
                
                self.stdout.write(f'\n📥 Downloading from your workspace: {workspace}...')
                
                save_dir = downloader.download_from_workspace(
                    workspace=workspace,
                    project='coffee-beans-defects',
                    version=1
                )
            
            # Count images in the downloaded dataset
            train_dir = os.path.join(save_dir, 'train', 'images')
            val_dir = os.path.join(save_dir, 'valid', 'images')
            
            train_count = 0
            val_count = 0
            
            if os.path.exists(train_dir):
                train_count = len([f for f in os.listdir(train_dir) 
                                 if f.endswith(('.jpg', '.png', '.jpeg'))])
            if os.path.exists(val_dir):
                val_count = len([f for f in os.listdir(val_dir) 
                               if f.endswith(('.jpg', '.png', '.jpeg'))])
            
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
                f'\n✅ Dataset downloaded successfully!\n'
                f'\n📁 Saved to: {save_dir}'
                f'\n📊 Training images: {train_count}'
                f'\n📊 Validation images: {val_count}'
                f'\n📊 Total images: {total_images}'
                f'\n🗄️  Database record ID: {dataset.id}'
                f'\n\nNext steps:'
                f'\n  1. Visit http://127.0.0.1:8000/defects/ to view dashboard'
                f'\n  2. Train the model from Django shell'
            ))
            
        except Exception as e:
            self.stderr.write(self.style.ERROR(
                f'\n❌ Download failed: {e}\n'
                f'\nTroubleshooting:'
                f'\n  1. Make sure you have access to the dataset'
                f'\n  2. Try forking it first at:'
                f'\n     https://universe.roboflow.com/technserve-labs/coffee-beans-defects'
                f'\n  3. Then download with:'
                f'\n     python manage.py download_defect_dataset --source workspace --workspace YOUR_USERNAME'
            ))