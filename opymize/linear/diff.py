
from opymize import Variable, LinOp
from opymize.linear.sparse import diffopn, lplcnop2

import numpy as np

try:
    import opymize.tools.gpu
    from pkg_resources import resource_stream
    from opymize.tools.gpu import prepare_kernels
except:
    # no cuda support
    pass

def imagedim_skips(imagedims):
    ndims = len(imagedims)
    skips = np.zeros(ndims, dtype=np.int64)
    skips[-1] = 1
    for t in range(ndims-2,-1,-1):
        skips[t] = skips[t+1]*imagedims[t+1]
    return skips

def staggered_diff_avgskips(imagedims):
    ndims = len(imagedims)
    skips = imagedim_skips(imagedims)
    navgskips =  1 << (ndims - 1)
    avgskips = np.zeros((ndims, navgskips), dtype=np.int64)
    coords = np.zeros(ndims, dtype=np.int64)
    for t in range(ndims):
        coords.fill(0)
        for m in range(navgskips):
            for c,s in zip(coords, skips):
                avgskips[t,m] += c*s
            for tt in range(ndims):
                if tt == t:
                    continue
                coords[tt] += 1
                if coords[tt] >= 2:
                    coords[tt] = 0
                else:
                    break
    return avgskips

def diff_prepare_gpu(imagedims, imageh, nchannels, weights, type_t):
    npoints = np.prod(imagedims)
    ndims = len(imagedims)
    skips = imagedim_skips(imagedims)
    constvars = {
        'GRAD_DIV': 1,
        'N': npoints, 'D': ndims, 'C': nchannels, 'dc_skip': ndims*nchannels,
        'skips': np.array(skips, dtype=np.int64, order='C'),
        'imagedims': np.array(imagedims, dtype=np.int64, order='C'),
        'avgskips': staggered_diff_avgskips(imagedims),
        'navgskips': 1 << (ndims - 1),
        'weights': weights,
        'imageh': imageh,
        'TYPE_T': type_t
    }
    files = [resource_stream('opymize.linear', 'diff.cu')]
    templates = [
        ("gradient", "PP", (npoints, nchannels, ndims), (24, 16, 2)),
        ("divergence", "PP", (npoints, nchannels, 1), (32, 24, 1)),
    ]
    return prepare_kernels(files, templates, constvars)

class GradientOp(LinOp):
    """ Gradients D x^k of x^k: y_t^k += b^k * D_t x^k`

    Example in two dimensions (+ are the grid points of x), k fixed:

           |                |
    ... -- + -- -- v1 -- -- + -- ...
           |                |
           |                |
           v2      Dx       w2
           |                |
           |                |
    ... -- + -- -- w1 -- -- + -- ...
           |                |

    v1 and w1 -- the partial x-derivatives at top and bottom
    v2 and w2 -- the partial y-derivatives at left and right
    Dx -- the gradient at the center of the box
    (Dx)_1 -- mean value of v1 and w1
    (Dx)_2 -- mean value of v2 and w2

    In one dimension there is no averaging, in three dimensions each
    derivative is the mean value of four finite differences etc.

    The "boundary" of y is left untouched.

    Note that the kernel of this operator is not only spanned by constant data,
    but by checkerboard patterns.
    """
    def __init__(self, imagedims, nchannels,
                       imageh=None, weights=None, adjoint=None):
        LinOp.__init__(self)
        ndims = len(imagedims)
        npoints = np.prod(imagedims)
        self.imagedims = imagedims
        self.nchannels = nchannels
        self.x = Variable((npoints, nchannels))
        self.y = Variable((npoints, ndims, nchannels))
        self.imageh = np.ones(ndims) if imageh is None else imageh
        self.weights = np.ones(nchannels) if weights is None else weights
        self._kernels = None
        self.spmat = diffopn(self.imagedims, components=self.nchannels,
                             steps=self.imageh, weights=self.weights,
                             schemes="centered")

        if adjoint is None:
            self.adjoint = DivergenceOp(imagedims, nchannels,
                imageh=self.imageh, weights=self.weights, adjoint=self)
        else:
            self.adjoint = adjoint

    def prepare_gpu(self, kernels=None, type_t="double"):
        if self._kernels is not None: return
        if kernels is None:
            kernels = diff_prepare_gpu(self.imagedims, self.imageh,
                                       self.nchannels, self.weights, type_t)
        self._kernels = kernels
        self.adjoint.prepare_gpu(kernels, type_t=type_t)

    def _call_gpu(self, x, y=None, add=False):
        assert y is not None
        if not add: y.fill(0.0)
        self._kernels['gradient'](x, y)

class DivergenceOp(LinOp):
    """ Negative divergence with Dirichlet boundary conditions

    Adjoint of GradientOp: x^k -= b^k * div(y^k)
    """
    def __init__(self, imagedims, nchannels,
                       imageh=None, weights=None, adjoint=None):
        LinOp.__init__(self)
        ndims = len(imagedims)
        npoints = np.prod(imagedims)
        self.imagedims = imagedims
        self.nchannels = nchannels
        self.x = Variable((npoints, ndims, nchannels))
        self.y = Variable((npoints, nchannels))
        self.imageh = np.ones(ndims) if imageh is None else imageh
        self.weights = np.ones(nchannels) if weights is None else weights
        self._kernels = None

        if adjoint is None:
            self.adjoint = GradientOp(imagedims, nchannels,
                imageh=self.imageh, weights=self.weights, adjoint=self)
        else:
            self.adjoint = adjoint

        self.spmat = self.adjoint.spmat.T

    def prepare_gpu(self, kernels=None, type_t="double"):
        if self._kernels is not None: return
        if kernels is None:
            kernels = diff_prepare_gpu(self.imagedims, self.imageh,
                                       self.nchannels, self.weights, type_t)
        self._kernels = kernels
        self.adjoint.prepare_gpu(kernels, type_t=type_t)

    def _call_gpu(self, x, y=None, add=False):
        assert y is not None
        if not add: y.fill(0.0)
        self._kernels['divergence'](x, y)

class LaplacianOp(LinOp):
    """ Laplacian operator with various boundary conditions """
    supported_bc = ["curvature", "curvature_adj",
                    "second-order", "second-order_adj",
                    "neumann",]
    def __init__(self, imagedims, nchannels, imageh=None,
                       boundary="neumann", adjoint=None):
        LinOp.__init__(self)
        npoints = np.prod(imagedims)
        ndims = len(imagedims)
        self.imagedims = imagedims
        self.imageh = np.ones(ndims) if imageh is None else imageh
        self.bc = boundary
        self.nchannels = nchannels
        self.x = Variable((npoints, nchannels))
        self.y = Variable((npoints, nchannels))
        self._kernel = None

        if self.bc[-4:] != "_adj":
            self.spmat = lplcnop2(self.imagedims, components=self.nchannels,
                                  steps=self.imageh, boundaries=self.bc)

        if self.bc == "neumann":
            self.adjoint = self
        elif self.bc in self.supported_bc:
            if adjoint is None:
                adj_bc = self.bc[:-4]
                if self.bc[-4:] != "_adj":
                    adj_bc = "%s_adj" % self.bc
                self.adjoint = LaplacianOp(imagedims, nchannels, imageh=imageh,
                                           boundary=adj_bc, adjoint=self)
            else:
                self.adjoint = adjoint
        else:
            raise Exception("Unknown boundary conditions: %s" % self.bc)

        if self.bc[-4:] == "_adj":
            self.spmat = self.adjoint.spmat.T

    def prepare_gpu(self, type_t="double"):
        if self._kernel is not None: return
        npoints = np.prod(self.imagedims)
        ndims = len(self.imagedims)
        skips = imagedim_skips(self.imagedims)
        constvars = {
            'LAPLACIAN': 1,
            'ADJOINT': 1 if self.bc[-4:] == "_adj" else 0,
            'boundary_conditions': self.bc[0].upper(),
            'N': npoints, 'D': ndims, 'C': self.nchannels,
            'skips': np.array(skips, dtype=np.int64, order='C'),
            'imagedims': np.array(self.imagedims, dtype=np.int64, order='C'),
            'imageh': self.imageh,
            'TYPE_T': type_t
        }
        files = [resource_stream('opymize.linear', 'diff.cu')]
        templates = [
            ("laplacian", "PP", (npoints, self.nchannels, 1), (32, 24, 1)),
        ]
        self._kernel = prepare_kernels(files, templates, constvars)['laplacian']
        self.adjoint.prepare_gpu(type_t=type_t)

    def _call_gpu(self, x, y=None, add=False):
        assert y is not None
        if not add: y.fill(0.0)
        self._kernel(x, y)
