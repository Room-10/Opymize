
import numpy as np

from opymize.functionals.l1norms import L1NormsConj
from opymize.tools.tests import checkOpDerivative

N = 23
L = 3
lbd = 1.3
tau = 0.7

F = L1NormsConj(N, (L, 1), lbd)
Fprox = F.prox(tau)

# Choose a point x0 where Fprox is smooth
x0 = np.random.randn(F.x.size)
x0 = F.x.vars(x0)[0]
norms = np.sqrt(np.einsum('itk->i', x0**2))
x0 *= (2.0 + 3.0*np.random.rand())*lbd/norms[:,None,None]

print("Testing Jacobian of L1 norm projection...")
checkOpDerivative(Fprox, x0.ravel())
