
import numpy as np

from opymize.functionals.ssd import SSD
from opymize.tools.tests import checkFctDerivative

N = 23
data = np.random.randn(N)
G = SSD(data[:,None])
x0 = np.random.randn(G.x.size)

print("Testing Gradient of SSD functional...")
checkFctDerivative(G, x0.ravel())
