import sys
import csv
import statistics
import pandas as pd
import statsmodels.api as sm
import scipy as sp
import lifelines as lf
import io as io
import matplotlib as mpl
import seaborn as sns

print(f"Python version: {sys.version}")
print(f"csv module version: {getattr(csv, '__version__', 'builtin')}")
print(f"statistics module version: {getattr(statistics, '__version__', 'builtin')}")
print(f"Pandas version: {pd.__version__}")
print(f"statsmodels.api version: {sm.__version__}")
print(f"SciPy version: {sp.__version__}")
print(f"lifelines version: {lf.__version__}")
print(f"io module version: {getattr(io, '__version__', 'builtin')}")
print(f"matplotlib version: {mpl.__version__}")
print(f"seaborn version: {sns.__version__}")