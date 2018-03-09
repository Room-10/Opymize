
import numpy as np
from numpy.linalg import norm

from opymize.linear import BlockOp, normest
from opymize.linear.scale import IdentityOp
from opymize.linear.diff import GradientOp
from opymize.functionals.ssd import SSD
from opymize.functionals.l1norms import L1Norms, L1NormsConj
from opymize.tools.tests import test_adjoint, test_rowwise_lp, \
    checkOpDerivative, checkFctDerivative

def main():
    imagedims = (10,12)
    l_labels = 3
    n_image = np.prod(imagedims)
    N = 23
    lbd = 0.3
    data = np.random.randn(n_image)

    print("=> Testing %s" % SSD)
    N = np.prod(imagedims)
    G = SSD(data[:,None])
    x0 = np.random.randn(G.x.size)
    checkFctDerivative(G, x0.ravel())

    grad = GradientOp(imagedims, l_labels)
    ident = IdentityOp(grad.y.size)
    linop = BlockOp([[grad,ident]])
    for op in [linop,linop.adjoint]:
        print("=> Testing %s" % type(op))
        print(normest(op))
        test_adjoint(op)
        test_rowwise_lp(op)

    lbd = 1.3
    tau = 0.7
    F = L1NormsConj(n_image, (l_labels, 1), lbd)
    Fprox = F.prox(tau)
    print("=> Testing %s" % type(Fprox))
    # Choose a point x0 where Fprox is smooth
    x0 = np.random.randn(F.x.size)
    x0 = F.x.vars(x0)[0]
    norms = np.sqrt(np.einsum('itk->i', x0**2))
    x0 *= (2.0 + 3.0*np.random.rand())*lbd/norms[:,None,None]
    checkOpDerivative(Fprox, x0.ravel())

if __name__ == "__main__":
    main()
