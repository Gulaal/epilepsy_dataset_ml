import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from selections.kfold import get_info
import matplotlib.pyplot as plt
import numpy as np

data = np.array(get_info())
lables = ['KFold Shuffle', 'KFold', 'Stratified KFold', 'Stratified KFold Shuffle']
bars = plt.bar([i for i in range(len(data))], data[:, 2], tick_label=lables)
plt.xticks(rotation=45)
plt.grid(True, linestyle='--')
plt.subplots_adjust(bottom=0.3)
plt.bar_label(bars, fmt='%.3f')
plt.show()