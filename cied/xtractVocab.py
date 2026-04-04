import pandas as pd
import pathlib
from fastai.vision.all import *

if platform.system() == 'Windows':
    pathlib.PosixPath = pathlib.WindowsPath
else:
    pathlib.WindowsPath = pathlib.PosixPath

def extract_vocabs(manuf_pkl, model_pkl):
    try:

        # 0. Extract segmentation Vocab
        #print(f"Opening {segment_pkl}...")
        #learn_segment = load_learner(segment_pkl)
        #segment_list = list(learn_segment.dls.vocab)
        #pd.DataFrame(segment_list, columns=['segment']).to_csv('cied/segmentation.csv', index=False)
        #print("Successfully saved: segmentation.csv")

        # 1. Extract Manufacturer Vocab
        print(f"Opening {manuf_pkl}...")
        learn_manuf = load_learner(manuf_pkl)
        manuf_list = list(learn_manuf.dls.vocab)
        pd.DataFrame(manuf_list, columns=['manufacturer']).to_csv('cied/manuf.csv', index=False)
        print("Successfully saved: manufacturer.csv")

        # 2. Extract Model Vocab
        print(f"Opening {model_pkl}...")
        learn_model = load_learner(model_pkl)
        model_list = list(learn_model.dls.vocab)
        pd.DataFrame(model_list, columns=['model_name']).to_csv('cied/model.csv', index=False)
        print("Successfully saved: model.csv")

    except Exception as e:
        print(f"Error: {e}")
        print("\nNote: Make sure all custom functions from the original CIED notebooks are defined in your current environment.")


# --- RUN THE SCRIPT ---
# Update these paths to where your .pkl files are actually located
extract_vocabs('cied/classification_manuf.pkl', 'cied/classification_model.pkl')