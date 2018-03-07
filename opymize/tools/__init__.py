
import numpy as np

def truncate(x, n):
    k = -int(np.floor(np.log10(abs(x))))
    # Example: x = 0.006142 => k = 3 / x = 2341.2 => k = -3
    k += n - 1
    if k > 0:
        x_str = str(abs(x))[:(k+2)]
    else:
        x_str = str(abs(x))[:n]+"0"*(-k)
    return np.sign(x)*float(x_str)