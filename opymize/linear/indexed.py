
from opymize import Variable, LinOp

import numba
import numpy as np
from numpy.linalg import norm

try:
    import opymize.tools.gpu
    from opymize.tools.gpu import prepare_kernels
    from pkg_resources import resource_stream
except:
    # no cuda support
    pass

def indexedmult_prepare_gpu(B, P, x, type_t="double"):
    J = B.shape[0]
    K = x[0]['shape'][1]
    L = B.shape[2]
    M = B.shape[1]
    N = x[0]['shape'][0]
    constvars = {
        'J': J, 'K': K, 'L': L, 'M': M, 'N': N,
        'P': P, 'B': B, 'TYPE_T': type_t
    }
    files = [resource_stream('opymize.linear', 'indexed.cu')]
    templates = [
        ("indexedmult", "PP", (N, J, M), (32, 24, 1)),
        ("indexedmultadj", "PP", (N, 1, 1), (512, 1, 1))
    ]
    return prepare_kernels(files, templates, constvars)

@numba.njit
def adjoint_multiply_indexed(y, P, B, x, precond=False):
    """ Does this: y[P] -= np.einsum('jml,jim->ijl', B, w)
    Unfortunately, advanced indexing without creating a copy is impossible.
    """
    for j in range(x.shape[0]):
        for i in range(x.shape[1]):
            for l in range(B.shape[2]):
                for m in range(x.shape[2]):
                    if precond:
                        y[i,P[j,l]] += np.abs(B[j,m,l])
                    else:
                        y[i,P[j,l]] -= B[j,m,l]*x[j,i,m]

class IndexedMultAdj(LinOp):
    """ for k,l,i do (Ax)[i,P[j,l]] -= \sum_m B[j,m,l] * x[j,i,m] """
    def __init__(self, K, N, P, B, adjoint=None):
        LinOp.__init__(self)
        assert P.shape[0] == B.shape[0]
        assert P.shape[1] == B.shape[2]
        self.x = Variable((B.shape[0],N,B.shape[1]))
        self.y = Variable((N,K))
        self.P = P
        self.B = B
        if adjoint is None:
            self.adjoint = IndexedMult(K, N, B, P, adjoint=self)
        else:
            self.adjoint = adjoint
        self._kernel = None

    def prepare_gpu(self, kernels=None, type_t="double"):
        if self._kernel is not None: return
        if kernels is None:
            kernels = indexedmult_prepare_gpu(self.B, self.P, self.y, type_t=type_t)
        self._kernel = kernels['indexedmultadj']
        self.adjoint.prepare_gpu(kernels, type_t=type_t)

    def _call_gpu(self, x, y=None, add=False):
        assert y is not None
        if not add: y.fill(0.0)
        self._kernel(x, y)

    def _call_cpu(self, x, y=None, add=False):
        assert y is not None
        if not add: y.fill(0.0)
        x = self.x.vars(x)[0]
        y = self.y.vars(y)[0]
        adjoint_multiply_indexed(y, self.P, self.B, x)

    def rowwise_lp(self, y, p=1, add=False):
        assert p is 1
        y = self.y.vars(y)[0]
        x = self.x.vars(self.x.new())[0]
        adjoint_multiply_indexed(y, self.P, self.B, x, precond=True)

class IndexedMult(LinOp):
    """ (Ax)[j,i,m] -= \sum_l B[j,m,l] * x[i,P[j,l]] """
    def __init__(self, K, N, B, P, adjoint=None):
        LinOp.__init__(self)
        assert P.shape[0] == B.shape[0]
        assert P.shape[1] == B.shape[2]
        self.x = Variable((N,K))
        self.y = Variable((B.shape[0],N,B.shape[1]))
        self.P = P
        self.B = B
        if adjoint is None:
            self.adjoint = IndexedMultAdj(K, N, P, B, adjoint=self)
        else:
            self.adjoint = adjoint
        self._kernel = None

    def prepare_gpu(self, kernels=None, type_t="double"):
        if self._kernel is not None: return
        if kernels is None:
            kernels = indexedmult_prepare_gpu(self.B, self.P, self.x, type_t=type_t)
        self._kernel = kernels['indexedmult']
        self.adjoint.prepare_gpu(kernels, type_t=type_t)

    def _call_gpu(self, x, y=None, add=False):
        assert y is not None
        if not add: y.fill(0.0)
        self._kernel(x, y)

    def _call_cpu(self, x, y=None, add=False):
        assert y is not None
        if not add: y.fill(0.0)
        x = self.x.vars(x)[0]
        y = self.y.vars(y)[0]
        y -= np.einsum('jml,ijl->jim', self.B, x[:,self.P])

    def rowwise_lp(self, y, p=1, add=False):
        assert p is 1
        assert add
        y = self.y.vars(y)[0]
        y += norm(self.B, ord=1, axis=2)[:,None,:]
