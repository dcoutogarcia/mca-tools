import numpy as np

def round_uncertainty(x):
    # The 1 turns 1 siginficant figure into 2 significant figures.
    if x != 0:
        return round(x, 1-int(np.floor(np.log10(abs(x)))))
    else:
        return 0


