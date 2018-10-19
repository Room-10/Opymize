
from opymize import Variable, LinOp

import numpy as np

try:
    import opymize.tools.gpu
    from pycuda import gpuarray
    from pycuda.elementwise import ElementwiseKernel
except:
    # no cuda support
    pass

class ZeroOp(LinOp):
    """ (Ax)_i = 0 """
    def __init__(self, M, N=None, adjoint=None):
        LinOp.__init__(self)
        N = M if N is None else N
        self.x = Variable(N)
        self.y = Variable(M)
        self.adjoint = ZeroOp(N, M, adjoint=self) if adjoint is None else adjoint
        self._call_cpu = self._call_gpu = self._call

    def _call(self, x, y=None, add=False):
        if not add:
            y = x if y is None else y
            y.fill(0.0)

    def rowwise_lp(self, y, p=1, add=False):
        if not add:
            y.fill(0.0)

class ScaleOp(LinOp):
    """ (Ax)_i = fact*x_i
        If fact is an array, do elementwise multiplication.
    """
    def __init__(self, N, fact):
        LinOp.__init__(self)
        self.x = Variable(N)
        self.y = Variable(N)
        self.adjoint = self
        self.fact = fact

    def prepare_gpu(self, type_t="double"):
        dtype = np.float64 if type_t == "double" else np.float32
        self.fact_gpu = gpuarray.to_gpu(np.asarray(self.fact, dtype=dtype))
        fact = "[i]" if type(self.fact) is np.ndarray else "[0]"
        self._kernel_add = ElementwiseKernel(
            "%s *x, %s *y, %s *fact" % (type_t, type_t, type_t),
            "y[i] += fact%s*x[i]" %  (fact,)
        )

    def _call_gpu(self, x, y=None, add=False):
        y = x if y is None else y
        if add:
            self._kernel_add(x, y, self.fact_gpu)
        else:
            y[:] = x
            y *= self.fact

    def _call_cpu(self, x, y=None, add=False):
        y = x if y is None else y
        if add:
            y += self.fact*x
        else:
            y[:] = x
            y *= self.fact

    def rowwise_lp(self, y, p=1, add=False):
        if add:
            y += np.abs(self.fact)**p
        else:
            y[:] = np.abs(self.fact)**p

class IdentityOp(LinOp):
    """ (Ax)_i = x_i """
    def __init__(self, N):
        LinOp.__init__(self)
        self.x = Variable(N)
        self.y = Variable(N)
        self.adjoint = self
        self._call_cpu = self._call_gpu = self._call

    def _call(self, x, y=None, add=False):
        if y is None and not add: return
        y = x if y is None else y
        if add:
            y += x
        else:
            y[:] = x

    def rowwise_lp(self, y, p=1, add=False):
        if add: y += 1.0
        else: y.fill(1.0)

class NihilOp(LinOp):
    """ (Ax)_i = x_i if keep_i==True else 0 """
    def __init__(self, N, keep):
        LinOp.__init__(self)
        self.x = Variable(N)
        self.y = Variable(N)
        self.adjoint = self
        self.keep = keep

    def _call_cpu(self, x, y=None, add=False):
        y = x if y is None else y
        if add:
            y[self.keep] += x[self.keep]
        else:
            y[np.logical_not(self.keep)] = 0.0
            y[self.keep] = x[self.keep]

    def rowwise_lp(self, y, p=1, add=False):
        if not add: y.fill(0.0)
        y[self.keep] += 1.0
