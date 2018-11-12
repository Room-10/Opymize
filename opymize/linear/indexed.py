
from opymize import Variable, LinOp
from opymize.linear import sparse as osp

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
        self.spmat = self.adjoint.spmat.T

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

        spP = osp.stackedop([osp.idxop(Pj, K) for Pj in P])
        spP = osp.extendedop(spP, before=(N,))
        spB = osp.extendedop(osp.diagop(B), before=(N,))
        spB = osp.transposeopn((N, B.shape[0], B.shape[1]), (1,0,2)).dot(spB)
        self.spmat = -spB.dot(spP)

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
