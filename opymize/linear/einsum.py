
from opymize import Variable, LinOp
from opymize.linear.sparse import einsumop

import numpy as np

try:
    import opymize.tools.gpu
    from opymize.tools.gpu import prepare_kernels
    from pkg_resources import resource_stream
    from pycuda import gpuarray
    from pycuda.elementwise import ElementwiseKernel
except:
    # no cuda support
    pass

class MatrixMult(LinOp):
    """ Matrix multiplication from the left:
        y_ji = \sum_k A_jk * x_ki
    """
    def __init__(self, N, A, trans=False, adjoint=None):
        LinOp.__init__(self)
        (j,k) = (1,0) if trans else (0,1)
        self.x = Variable((A.shape[j],N))
        self.y = Variable((A.shape[k],N))
        self.trans = trans
        self.A = A
        if adjoint is None:
            self.adjoint = MatrixMult(N, A, trans=not trans, adjoint=self)
        else:
            self.adjoint = adjoint
        self._kernel = None

    def prepare_gpu(self, type_t="double"):
        if self._kernel is not None: return
        K, N = self.x[0]['shape']
        J, _ = self.y[0]['shape']
        constvars = {
            'A': self.A,
            'J': J, 'K': K, 'N': N,
            'trans': 't' if self.trans else 'n',
            'MATRIX_MULT': 1,
            'TYPE_T': type_t
        }
        files = [resource_stream('opymize.linear', 'einsum.cu')]
        templates = [("matrixmult", "PP", (J, N, 1), (32, 24, 1))]
        self._kernel = prepare_kernels(files, templates, constvars)['matrixmult']
        self.adjoint.prepare_gpu(type_t=type_t)

    def _call_gpu(self, x, y=None, add=False):
        assert y is not None
        if not add: y.fill(0.0)
        self._kernel(x, y)

    def _call_cpu(self, x, y=None, add=False):
        x = self.x.vars(x)[0]
        y = self.y.vars(y)[0] if y is not None else x
        if not add: y.fill(0.0)
        descr = 'kj,ki->ji' if self.trans else 'jk,ki->ji'
        y += np.einsum(descr, self.A, x)

    def rowwise_lp(self, y, p=1, add=False):
        y = self.y.vars(y)[0]
        # uses broadcasting
        A = self.A.T if self.trans else self.A
        if add:
            y += np.sum(np.abs(A)**p, axis=1)[:,None]
        else:
            y[:] = np.sum(np.abs(A)**p, axis=1)[:,None]

class MatrixMultR(LinOp):
    """ Matrix multiplication from the right:
        y_ij = \sum_k x_ik * A_kj
    """
    def __init__(self, N, A, trans=False, adjoint=None):
        LinOp.__init__(self)
        (k,j) = (1,0) if trans else (0,1)
        self.x = Variable((N,A.shape[k]))
        self.y = Variable((N,A.shape[j]))
        self.trans = trans
        self.A = A
        if adjoint is None:
            self.adjoint = MatrixMultR(N, A, trans=not trans, adjoint=self)
        else:
            self.adjoint = adjoint
        self._kernel = None

    def prepare_gpu(self, type_t="double"):
        if self._kernel is not None: return
        N, K = self.x[0]['shape']
        _, J = self.y[0]['shape']
        constvars = {
            'A': self.A,
            'J': J, 'K': K, 'N': N,
            'trans': 't' if self.trans else 'n',
            'MATRIX_MULT_R': 1,
            'TYPE_T': type_t
        }
        files = [resource_stream('opymize.linear', 'einsum.cu')]
        templates = [("matrixmultr", "PP", (N, J, 1), (32, 24, 1))]
        self._kernel = prepare_kernels(files, templates, constvars)['matrixmultr']
        self.adjoint.prepare_gpu(type_t=type_t)

    def _call_gpu(self, x, y=None, add=False):
        assert y is not None
        if not add: y.fill(0.0)
        self._kernel(x, y)

    def _call_cpu(self, x, y=None, add=False):
        x = self.x.vars(x)[0]
        y = self.y.vars(y)[0] if y is not None else x
        if not add: y.fill(0.0)
        descr = 'ik,jk->ij' if self.trans else 'ik,kj->ij'
        y += np.einsum(descr, x, self.A)

    def rowwise_lp(self, y, p=1, add=False):
        y = self.y.vars(y)[0]
        # uses broadcasting
        A = self.A.T if self.trans else self.A
        if add:
            y += np.sum(np.abs(A)**p, axis=0)[None,:]
        else:
            y[:] = np.sum(np.abs(A)**p, axis=0)[None,:]

class MatrixOp(LinOp):
    """ y_i = \sum_j A_ij * x_j """
    def __init__(self, A):
        MatrixMult.__init__(self, 1, A)

class DiagMatrixMultR(LinOp):
    """ Matrix multiplication from the right by diagonal matrix:
        y_ik =  x_ik * A_k
    """
    def __init__(self, N, A):
        LinOp.__init__(self)
        self.x = Variable((N,A.size))
        self.y = Variable((N,A.size))
        self.A = A
        self.adjoint = self
        self._kernel = None
        self.A_gpu = None

    def prepare_gpu(self, type_t="double"):
        self.A_gpu = gpuarray.to_gpu(self.A)
        self._kernel_add = ElementwiseKernel(
            "%s *x, %s *y, %s *A" % (type_t,type_t,type_t,),
            "y[i] += x[i]*A[i %% %d]" % self.A.size
        )

    def _call_gpu(self, x, y=None, add=False):
        assert y is not None
        if not add: y.fill(0.0)
        self._kernel_add(x, y, self.A_gpu)

    def _call_cpu(self, x, y=None, add=False):
        x = self.x.vars(x)[0]
        y = self.y.vars(y)[0] if y is not None else x
        if not add: y.fill(0.0)
        y += np.einsum('ik,k->ik', x, self.A)

    def rowwise_lp(self, y, p=1, add=False):
        y = self.y.vars(y)[0]
        if not add: y.fill(0.0)
        y += np.abs(self.A)[None,:]

class TangledMatrixMultR(LinOp):
    """ Entangled matrix multiplication from the right:
        y_mik = \sum_jl x_jil * A_jlmk

    This is called entangled matrix multiplication, because a transposition and
    combination of axes would yield a usual matrix multiplication:
        y_(i)(mk) = \sum_(jl) x_(i)(jl) * A_(jl)(mk)
    """
    def __init__(self, N, A, trans=False, adjoint=None):
        LinOp.__init__(self)
        (j,m) = (2,0) if trans else (0,2)
        self.x = Variable((A.shape[j],N,A.shape[j+1]))
        self.y = Variable((A.shape[m],N,A.shape[m+1]))
        self.trans = trans
        self.A = A
        if adjoint is None:
            subscripts =  'mkjl,jil->mik' if self.trans else 'jlmk,jil->mik'
            self.spmat = einsumop(subscripts, self.A, self.x[0]['shape'])
            self.adjoint = TangledMatrixMultR(N, A, trans=not trans, adjoint=self)
        else:
            self.adjoint = adjoint
            self.spmat = self.adjoint.spmat.T
        self._kernel = None

    def prepare_gpu(self, type_t="double"):
        if self._kernel is not None: return
        J, N, L = self.x[0]['shape']
        M, _, K = self.y[0]['shape']
        constvars = {
            'A': self.A,
            'J': J, 'K': K, 'L': L, 'M': M, 'N': N, 'MK': M*K,
            'trans': 't' if self.trans else 'n',
            'TANGLED_MATRIX_MULT_R': 1,
            'TYPE_T': type_t
        }
        files = [resource_stream('opymize.linear', 'einsum.cu')]
        templates = [("tangledmatrixmultr", "PP", (N, M*K, 1), (32, 24, 1))]
        self._kernel = prepare_kernels(files, templates, constvars)['tangledmatrixmultr']
        self.adjoint.prepare_gpu(type_t=type_t)

    def _call_gpu(self, x, y=None, add=False):
        assert y is not None
        if not add: y.fill(0.0)
        self._kernel(x, y)

class MatrixMultRBatched(LinOp):
    """ Batched matrix multiplication from the right:
        y_jlk = \sum_m x_jlm * A_jmk
    """
    def __init__(self, N, A, trans=False, adjoint=None):
        LinOp.__init__(self)
        (m,k) = (2,1) if trans else (1,2)
        self.x = Variable((A.shape[0],N,A.shape[m]))
        self.y = Variable((A.shape[0],N,A.shape[k]))
        self.trans = trans
        self.A = A
        if adjoint is None:
            subscripts = 'jkm,jlm->jlk' if self.trans else 'jmk,jlm->jlk'
            self.spmat = einsumop(subscripts, self.A, self.x[0]['shape'])
            self.adjoint = MatrixMultRBatched(N, A, trans=not trans, adjoint=self)
        else:
            self.adjoint = adjoint
            self.spmat = self.adjoint.spmat.T
        self._kernel = None

    def prepare_gpu(self, type_t="double"):
        if self._kernel is not None: return
        J, L, K = self.y[0]['shape']
        M = self.x[0]['shape'][2]
        constvars = {
            'A': self.A,
            'J': J, 'K': K, 'L': L, 'M': M,
            'trans': 't' if self.trans else 'n',
            'MATRIX_MULT_R_BATCHED': 1,
            'TYPE_T': type_t
        }
        files = [resource_stream('opymize.linear', 'einsum.cu')]
        templates = [("matrixmultrbatched", "PP", (J, L, K), (8, 16, 4))]
        self._kernel = prepare_kernels(files, templates, constvars)['matrixmultrbatched']
        self.adjoint.prepare_gpu(type_t=type_t)

    def _call_gpu(self, x, y=None, add=False):
        assert y is not None
        if not add: y.fill(0.0)
        self._kernel(x, y)
