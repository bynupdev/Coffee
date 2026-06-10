"""Downloads datasets from Roboflow"""
import os
import requests
import zipfile
from pathlib import Path
from django.conf import settings

class RoboflowDownloader:
    def __init__(self, api_key=None):
        self.api_key = api_key or os.environ.get('ROBOFLOW_API_KEY') or getattr(settings, 'ROBOFLOW_API_KEY', None)
    
    def download_dataset(self, workspace, project, version, save_dir=None):
        """Download a dataset from Roboflow"""
        if not self.api_key:
            raise ValueError("ROBOFLOW_API_KEY is not set")
        
        # CORRECTED: Use the proper API endpoint structure
        # For Roboflow Universe datasets, use this format:
        url = f"https://api.roboflow.com/v1/universe/{workspace}/{project}/dataset/{version}/yolov8"
        
        print(f"Downloading from Roboflow Universe...")
        print(f"Workspace: {workspace}")
        print(f"Project: {project}")
        print(f"Version: {version}")
        print("This may take a few minutes...")
        
        # Make the request with proper parameters
        params = {
            'api_key': self.api_key
        }
        
        response = requests.get(url, params=params)
        
        # Debug: Print response info if there's an error
        if response.status_code != 200:
            print(f"Response status: {response.status_code}")
            print(f"Response content: {response.text[:500]}")
            raise Exception(
                f"Download failed with status {response.status_code}. "
                f"Make sure you have access to this dataset. "
                f"Try visiting: https://universe.roboflow.com/{workspace}/{project}"
            )
        
        # Set save directory
        if save_dir is None:
            base = Path(settings.BASE_DIR) / 'defect_detector' / 'datasets' / 'roboflow'
            save_dir = base / project
        
        os.makedirs(save_dir, exist_ok=True)
        
        # Save and extract
        zip_path = os.path.join(save_dir, 'dataset.zip')
        with open(zip_path, 'wb') as f:
            f.write(response.content)
        
        # Verify it's a valid zip file
        if not zipfile.is_zipfile(zip_path):
            os.remove(zip_path)
            raise Exception(
                "Downloaded file is not a valid zip. "
                "You may need to fork this dataset to your workspace first. "
                f"Visit: https://universe.roboflow.com/{workspace}/{project}"
            )
        
        # Extract
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(save_dir)
        
        # Remove zip
        os.remove(zip_path)
        
        print(f"Dataset extracted to: {save_dir}")
        return str(save_dir)
    
    def download_from_workspace(self, workspace, project, version, save_dir=None):
        """Alternative: Download from your own workspace (after forking)"""
        if not self.api_key:
            raise ValueError("ROBOFLOW_API_KEY is not set")
        
        # For datasets in your own workspace
        url = f"https://api.roboflow.com/v1/{workspace}/{project}/{version}/yolov8"
        
        print(f"Downloading from your workspace...")
        print(f"Workspace: {workspace}")
        print(f"Project: {project}")
        
        params = {'api_key': self.api_key}
        response = requests.get(url, params=params)
        
        if response.status_code != 200:
            raise Exception(f"Download failed: {response.status_code} - {response.text[:200]}")
        
        if save_dir is None:
            base = Path(settings.BASE_DIR) / 'defect_detector' / 'datasets' / 'roboflow'
            save_dir = base / project
        
        os.makedirs(save_dir, exist_ok=True)
        
        zip_path = os.path.join(save_dir, 'dataset.zip')
        with open(zip_path, 'wb') as f:
            f.write(response.content)
        
        if not zipfile.is_zipfile(zip_path):
            os.remove(zip_path)
            raise Exception("Downloaded file is not a valid zip")
        
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(save_dir)
        
        os.remove(zip_path)
        
        print(f"Dataset extracted to: {save_dir}")
        return str(save_dir)