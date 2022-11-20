import os
import os.path as op
import matplotlib.pyplot as plt
from IPython.display import set_matplotlib_formats
import ndslib.templates as tt
import warnings
import pandas as pd

def jupyter_startup():
    """
    Configure the Jupyter notebook to have the right figure size and style.

    Also, suppress warnings.
    """
    warnings.filterwarnings('ignore')
    warnings.filterwarnings('ignore', category=FutureWarning)
    warnings.filterwarnings('ignore', category=pd.errors.PerformanceWarning)
    if os.environ.get('NDS_PDF'):
        plt.rcParams["figure.figsize"] = 4.56, 3.42
        plt.rcParams['figure.dpi'] = 300
        plt.rcParams['font.size'] = 8
        plt.style.use(op.join(tt.__path__[0], "nds.mplstyle"))
    if os.environ.get('NDS_SVG'):
        set_matplotlib_formats('svg')
    else:
        set_matplotlib_formats('png')
