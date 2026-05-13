import pathlib
import platform
from fastai.vision.all import *

# 3. Patch for Windows/Linux Path compatibility
if platform.system() == 'Windows':
    pathlib.PosixPath = pathlib.WindowsPath
else:
    pathlib.WindowsPath = pathlib.PosixPath

# 1. Load the exported learner
# Replace 'Zhukov.pkl' with the actual filename/path of your file
learn_inf = load_learner('cied\classification_manuf.pkl')

# 2. Extract the list of manufacturers (the classes)
manufacturers = learn_inf.dls.vocab

# 3. Print the results
print(f"Total number of manufacturers: {len(manufacturers)}")
print("--- List of Classes ---")
for i, name in enumerate(manufacturers):
    print(f"{i}: {name}")