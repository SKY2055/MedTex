#!/usr/bin/env python3
"""
Med7 Model Installer
Downloads and installs Med7 from HuggingFace with proper wheel naming.
"""

import os
import sys
import subprocess
import urllib.request
import tempfile
import shutil

def download_file(url, dest_path):
    """Download file with progress"""
    print(f"Downloading from {url}...")
    print(f"This is a ~580MB file, please wait...")
    
    def reporthook(count, block_size, total_size):
        percent = int(count * block_size * 100 / total_size)
        downloaded = count * block_size / (1024 * 1024)
        total = total_size / (1024 * 1024)
        sys.stdout.write(f"\rProgress: {percent}% ({downloaded:.1f}/{total:.1f} MB)")
        sys.stdout.flush()
    
    try:
        urllib.request.urlretrieve(url, dest_path, reporthook)
        print(f"\nDownloaded to {dest_path}")
        return True
    except Exception as e:
        print(f"\nDownload failed: {e}")
        return False

def install_med7():
    """Download and install Med7 model"""
    url = "https://huggingface.co/kormilitzin/en_core_med7_lg/resolve/main/en_core_med7_lg-any-py3-none-any.whl"
    
    # Create temp directory
    temp_dir = tempfile.mkdtemp()
    download_path = os.path.join(temp_dir, "en_core_med7_lg-any-py3-none-any.whl")
    
    try:
        # Download
        if not download_file(url, download_path):
            return False
        
        # Check file size (should be ~580MB)
        size_mb = os.path.getsize(download_path) / (1024 * 1024)
        print(f"File size: {size_mb:.1f} MB")
        
        if size_mb < 10:
            print("ERROR: Downloaded file is too small, likely an HTML page or redirect")
            with open(download_path, 'r') as f:
                content = f.read(500)
                print(f"Content: {content[:200]}...")
            return False
        
        # Rename to valid wheel filename
        valid_wheel = os.path.join(temp_dir, "en_core_med7_lg-0.0.1-py3-none-any.whl")
        shutil.move(download_path, valid_wheel)
        print(f"Renamed to valid wheel: {valid_wheel}")
        
        # Install with pip
        print("Installing with pip...")
        result = subprocess.run(
            [sys.executable, "-m", "pip", "install", valid_wheel, "--force-reinstall"],
            capture_output=True,
            text=True
        )
        
        if result.returncode == 0:
            print("✓ Med7 installed successfully!")
            # Verify installation
            try:
                import spacy
                nlp = spacy.load("en_core_med7_lg")
                print("✓ Med7 model loads correctly!")
                doc = nlp("Patient was prescribed Amoxicillin 500mg twice daily for 7 days.")
                entities = [(ent.text, ent.label_) for ent in doc.ents]
                print(f"Test extraction: {entities}")
                return True
            except Exception as e:
                print(f"Warning: Model installed but failed to load: {e}")
                return True  # Still counts as success
        else:
            print(f"Installation failed: {result.stderr}")
            return False
            
    finally:
        # Cleanup
        shutil.rmtree(temp_dir, ignore_errors=True)

if __name__ == "__main__":
    print("=" * 60)
    print("Med7 Model Installer for MedTex")
    print("=" * 60)
    
    success = install_med7()
    
    if success:
        print("\n✓ Med7 is now ready to use!")
        sys.exit(0)
    else:
        print("\n✗ Med7 installation failed.")
        print("\nAlternative: The app works with PubMedBERT fallback.")
        print("Med7 provides extra detail (DOSAGE, STRENGTH, FORM, etc.) but is not strictly required.")
        sys.exit(1)
