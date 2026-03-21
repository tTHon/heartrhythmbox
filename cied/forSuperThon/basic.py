#old model no segment, one at a time
import sys
import pathlib
import platform
import numpy as np
import traceback

# ==========================================================
# STEP 1: COMPATIBILITY PATCHES (MUST BE AT THE VERY TOP)
# ==========================================================

# 1. Patch for NumPy 'int' attribute (NumPy 2.0+ compatibility)
if not hasattr(np, 'int'):
    np.int = int

# 2. Patch for 'AMPMode' (FastAI version bridge)
import fastai.callback.fp16
if not hasattr(fastai.callback.fp16, 'AMPMode'):
    class AMPMode:
        def __init__(self, *args, **kwargs): pass 
    fastai.callback.fp16.AMPMode = AMPMode
    sys.modules['fastai.callback.fp16'].AMPMode = AMPMode

# 3. Patch for Windows/Linux Path compatibility
if platform.system() == 'Windows':
    pathlib.PosixPath = pathlib.WindowsPath
else:
    pathlib.WindowsPath = pathlib.PosixPath

from fastai.vision.all import *

# ==========================================================
# STEP 2: CONFIGURATION
# ==========================================================
# Use forward slashes or pathlib to ensure Windows compatibility
MODEL_DIR  = pathlib.Path('cied')
IMAGE_PATH = MODEL_DIR / 'Dataset' / 'A3.png'

FILES = {
    'manuf': MODEL_DIR / 'classification_manuf.pkl',
    'model': MODEL_DIR / 'classification_model.pkl'
}

# ==========================================================
# STEP 3: MAIN EXECUTION
# ==========================================================
def run_inference():
    try:
        print("--- CIED Identification System ---")
        
        # Check if files exist before loading
        for key, path in FILES.items():
            if not path.exists():
                raise FileNotFoundError(f"Missing model file: {path}")
        if not IMAGE_PATH.exists():
            raise FileNotFoundError(f"Missing image file: {IMAGE_PATH}")

        # Load models
        print("Loading AI Models (CPU Mode)...")
        learn_manuf = load_learner(FILES['manuf'], cpu=True)
        learn_model = load_learner(FILES['model'], cpu=True)

        print(f"Analyzing: {IMAGE_PATH.name}...")

        # PREDICTION: Passing the string path is the safest way to trigger 
        # the model's internal resizing and normalization transforms.
        res_manuf = learn_manuf.predict(str(IMAGE_PATH))
        res_model = learn_model.predict(str(IMAGE_PATH))

        # Data Extraction
        # predict returns: (Class_Name, Class_Index, Probabilities)
        name_m, _, probs_m = res_manuf
        name_g, _, probs_g = res_model

        # Calculate confidence percentages
        conf_m = probs_m.max().item() * 100
        conf_g = probs_g.max().item() * 100

        # FINAL OUTPUT
        print("\n" + "═"*45)
        print(f" DETECTION RESULTS")
        print("─"*45)
        print(f" MANUFACTURER : {name_m:<20} [Confidence: {conf_m:.2f}%]")
        print(f" MODEL GROUP  : {name_g:<20} [Confidence: {conf_g:.2f}%]")
        print("═"*45)

    except Exception as e:
        print(f"\n[ERROR]: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    run_inference()