import sys
import statistics
import pandas as pd
import statsmodels.api as sm
import scipy as sp
import lifelines as lf
import matplotlib as mpl
import seaborn as sns
import numpy as np

print(f"Python version: {sys.version}")
print(f"statistics module version: {getattr(statistics, '__version__', 'builtin')}")
print(f"Pandas version: {pd.__version__}")
print(f"statsmodels.api version: {sm.__version__}")
print(f"SciPy version: {sp.__version__}")
print(f"lifelines version: {lf.__version__}")
print(f"matplotlib version: {mpl.__version__}")
print(f"seaborn version: {sns.__version__}")
print(f"NumPy version: {np.__version__}")