import os
import shutil
import bitsandbytes as bnb
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)

def fix_bitsandbytes():
    """
    Attempts to fix the bitsandbytes installation by creating a symlink/copy 
    for the missing CUDA library (e.g., libbitsandbytes_cuda128.so).
    """
    try:
        # Locate the package directory
        if hasattr(bnb, '__path__'):
            package_dir = Path(bnb.__path__[0])
        else:
            package_dir = Path(bnb.__file__).parent
            
        logging.info(f"Bitsandbytes directory: {package_dir}")
        
        # The warning usually complains about a missing file ending in _cudaXYZ.so
        # We will look for the highest available version and copy it to the expected one.
        # Since we don't know exactly what 12.8 expects (likely cuda128), we will try to find it.
        
        # Check what files exist
        so_files = list(package_dir.glob("libbitsandbytes_cuda*.so"))
        if not so_files:
            logging.error("No CUDA .so files found in bitsandbytes directory!")
            return

        logging.info(f"Found existing libraries: {[f.name for f in so_files]}")

        # We need to handle the specific error: "Could not find ... libbitsandbytes_cuda128.so"
        # So we target that filename.
        target_filename = "libbitsandbytes_cuda128.so"
        target_path = package_dir / target_filename
        
        if target_path.exists():
            logging.info(f"Target file {target_filename} already exists.")
            # It might exist but be corrupt or wrong, but let's assume if it's there, we leave it.
            # Or maybe we should overwrite? Let's check if it's 0 bytes.
            if target_path.stat().st_size > 0:
                logging.info("File seems valid. Skipping patch.")
                return

        # Find best candidate: prefer cuda12x, then cuda11x
        # Sort by name descending to get higher numbers first
        candidates = sorted(so_files, key=lambda x: x.name, reverse=True)
        best_candidate = candidates[0]
        
        logging.info(f"Selected fallback source: {best_candidate.name}")
        logging.info(f"Copying {best_candidate.name} -> {target_filename}")
        
        shutil.copy2(best_candidate, target_path)
        logging.info("Successfully patched bitsandbytes!")
        logging.info("Please run your training script again.")

    except Exception as e:
        logging.error(f"Failed to apply fix: {e}")

if __name__ == "__main__":
    fix_bitsandbytes()
