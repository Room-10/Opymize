
"""
Set of functions to compute the gradients D x^k of x^k (or its adjoint operator)
on a staggered grid and setting `y_t^k += b^k * D_t x^k` (or
`x^k -= b^k * div(y^k)`). The "boundary" of y is left untouched.

Example in two dimensions (+ are the grid points of x), k fixed:

       |                |
... -- + -- -- v1 -- -- + -- ...
       |                |          v1 and w1 -- the partial x-derivatives at top and bottom
       |                |          v2 and w2 -- the partial y-derivatives at left and right
       v2      Dx       w2         Dx -- the gradient at the center of the box
       |                |          (Dx)_1 -- mean value of v1 and w1
       |                |          (Dx)_2 -- mean value of v2 and w2
... -- + -- -- w1 -- -- + -- ...
       |                |

In one dimension there is no averaging, in three dimensions each
derivative is the mean value of four finite differences etc.

The adjoint operator (reading from y and writing to x) is understood to be the
negative divergence -div(y^k) with Dirichlet boundary.
"""

from opymize import Variable, LinOp

import numpy as np
import itertools
import numba

try:
    import opymize.tools.gpu
    from pkg_resources import resource_stream
    from opymize.tools.gpu import prepare_kernels
except:
    # no cuda support
    pass

def staggered_diff_avgskips(imagedims):
    D = len(imagedims)
    skips = tuple(int(np.prod(imagedims[(D-t):])) for t in range(D))
    navgskips =  1 << (D - 1)
    avgskips = np.zeros([D, navgskips], dtype=np.int64, order='C')
    for t in range(D):
        for m, p in enumerate(itertools.product([0, 1], repeat=(D - 1))):
            avgskips[t,m] = np.inner(p[:t] + (0,) + p[t:], skips)
    return avgskips

@numba.njit
def gradient(x, y, b, avgskips, imagedims, precond=False):
    staggered_diff(x, y, b, avgskips, imagedims, adjoint=False, precond=precond)

@numba.njit
def divergence(x, y, b, avgskips, imagedims, precond=False):
    staggered_diff(y, x, b, avgskips, imagedims, adjoint=True, precond=precond)

@numba.njit
def staggered_diff(x, y, b, avgskips, imagedims, adjoint=False, precond=False):
    """ Computes `y[i,t,k] += b[k] * D[t,i,:]x[:,k]`.

    N = prod(imagedims)
    D = len(imagedims)
    C = x.shape[1] (number of channels)

    Args:
        x : function to be derived, shape (N,C)
        y : gradient (when `adjoint == False`), shape (N,D,C)
        b : weights for the features/channels of x
        avgskips : output of staggered_diff_avgskips(imagedims)
        imagedims : tuple, shape of the image domain
        adjoint : (optional) if True, apply the adjoint operator (reading from y
                  and writing to x), i.e. `x[i,k] -= b[k] * div[i,:,:]y[:,:,k]`.
        precond : Computes rowwise/colwise L1 norm of `diag(b)D`.
    """
    N, D, C = y.shape
    navgskips =  1 << (D - 1)

    skips = np.zeros(D, dtype=np.int64)
    skips[0] = 1
    for t in range(1, D):
        skips[t] = skips[t-1]*imagedims[D-t]

    coords = np.zeros(D, dtype=np.int64)

    for k in range(C):
        for t in range(D):
            coords *= 0
            for i in range(N):
                # ignore boundary points
                in_range = True
                dc = D
                while dc > 0:
                    dc -= 1
                    if coords[dc] >= imagedims[dc] - 1:
                        in_range = False
                        break

                if in_range:
                    # regular case
                    bk = b[k]/navgskips

                    for avgskip in avgskips[t]:
                        base = i + avgskip
                        if precond:
                            if adjoint:
                                x[base + skips[t],k] += np.abs(bk)
                                x[base,k] += np.abs(bk)
                            else:
                                y[i,t,k] += 2*np.abs(bk)
                        else:
                            if adjoint:
                                x[base + skips[t],k] += bk * y[i,t,k]
                                x[base,k] -= bk * y[i,t,k]
                            else:
                                y[i,t,k] += bk * x[base + skips[t],k]
                                y[i,t,k] -= bk * x[base,k]

                # advance coordinates
                dd = D
                while dd > 0:
                    dd -= 1
                    coords[dd] += 1
                    if coords[dd] >= imagedims[dd]:
                        coords[dd] = 0
                    else:
                        break

def diff_prepare_gpu(imagedims, C, weights, type_t="double"):
    N = np.prod(imagedims)
    D = len(imagedims)
    skips = (1,)
    for d in reversed(imagedims[1:]):
        skips += (skips[-1]*d,)
    constvars = {
        'N': N, 'D': D, 'C': C, 'dc_skip': D*C,
        'skips': np.array(skips, dtype=np.int64, order='C'),
        'imagedims': np.array(imagedims, dtype=np.int64, order='C'),
        'avgskips': staggered_diff_avgskips(imagedims),
        'navgskips': 1 << (D - 1),
        'weights': weights,
        'TYPE_T': type_t
    }
    files = [resource_stream('opymize.linear', 'diff.cu')]
    fd = 4 if type_t == "double" else 5
    templates = [
        ("gradient", "PP", (N, C, D), (fd*6, 16, 2)),
        ("divergence", "PP", (N, C, 1), (fd*8, 24, 1)),
    ]
    return prepare_kernels(files, templates, constvars)

class GradientOp(LinOp):
    def __init__(self, imagedims, C, weights=None, adjoint=None):
        LinOp.__init__(self)
        D = len(imagedims)
        N = np.prod(imagedims)
        self.imagedims = imagedims
        self.C = C
        self.x = Variable((N, C))
        self.y = Variable((N, D, C))
        self.avgskips = staggered_diff_avgskips(imagedims)
        self.weights = np.ones(C) if weights is None else weights
        self._kernels = None

        if adjoint is None:
            self.adjoint = DivergenceOp(imagedims, C,
                weights=self.weights, adjoint=self)
        else:
            self.adjoint = adjoint

    def prepare_gpu(self, kernels=None, type_t="double"):
        if self._kernels is not None: return
        if kernels is None:
            kernels = diff_prepare_gpu(self.imagedims, self.C, self.weights,
                                       type_t=type_t)
        self._kernels = kernels
        self.adjoint.prepare_gpu(kernels, type_t=type_t)

    def _call_gpu(self, x, y=None, add=False):
        assert y is not None
        if not add: y.fill(0.0)
        self._kernels['gradient'](x, y)

    def _call_cpu(self, x, y=None, add=False):
        assert y is not None
        x = self.x.vars(x)[0]
        y = self.y.vars(y)[0]
        if not add: y.fill(0.0)
        gradient(x, y, self.weights, self.avgskips, self.imagedims)

    def rowwise_lp(self, y, p=1, add=False):
        assert(p == 1)
        y = self.y.vars(y)[0]
        x = np.empty(self.x[0]['shape'])
        if not add: y.fill(0.0)
        gradient(x, y, self.weights, self.avgskips, self.imagedims, precond=True)

class DivergenceOp(LinOp):
    def __init__(self, imagedims, C, weights=None, adjoint=None):
        LinOp.__init__(self)
        D = len(imagedims)
        N = np.prod(imagedims)
        self.imagedims = imagedims
        self.C = C
        self.x = Variable((N, D, C))
        self.y = Variable((N, C))
        self.avgskips = staggered_diff_avgskips(imagedims)
        self.weights = np.ones(C) if weights is None else weights
        self._kernels = None

        if adjoint is None:
            self.adjoint = GradientOp(imagedims, C,
                weights=self.weights, adjoint=self)
        else:
            self.adjoint = adjoint

    def prepare_gpu(self, kernels=None, type_t="double"):
        if self._kernels is not None: return
        if kernels is None:
            kernels = diff_prepare_gpu(self.imagedims, self.C, self.weights,
                                       type_t=type_t)
        self._kernels = kernels
        self.adjoint.prepare_gpu(kernels, type_t=type_t)

    def _call_gpu(self, x, y=None, add=False):
        assert y is not None
        if not add: y.fill(0.0)
        self._kernels['divergence'](x, y)

    def _call_cpu(self, x, y=None, add=False):
        assert y is not None
        x = self.x.vars(x)[0]
        y = self.y.vars(y)[0]
        if not add: y.fill(0.0)
        divergence(x, y, self.weights, self.avgskips, self.imagedims)

    def rowwise_lp(self, y, p=1, add=False):
        assert(p == 1)
        y = self.y.vars(y)[0]
        x = np.empty(self.x[0]['shape'])
        if not add: y.fill(0.0)
        divergence(x, y, self.weights, self.avgskips, self.imagedims, precond=True)
