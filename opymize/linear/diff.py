
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

@numba.njit
def imagedim_skips(imagedims):
    ndims = len(imagedims)
    skips = np.zeros(ndims, dtype=np.int64)
    skips[-1] = 1
    for t in range(ndims-2,-1,-1):
        skips[t] = skips[t+1]*imagedims[t+1]
    return skips

@numba.njit
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

@numba.njit
def gradient(x, y, imagedims, imageh, weights, precond=False):
    staggered_diff(x, y, imagedims, imageh, weights, False, precond)

@numba.njit
def divergence(x, y, imagedims, imageh, weights, precond=False):
    staggered_diff(y, x, imagedims, imageh, weights, True, precond)

@numba.njit
def staggered_diff(x, y, imagedims, imageh, weights, adjoint, precond):
    """ Computes `y[i,t,k] += weights[k]*D[t,i,:]x[:,k]`.

    Args:
        x : ndarray of floats, shape (npoints, nchannels)
            function to be derived
        y : ndarray of floats, shape (npoints, ndims, nchannels)
            gradient (when `adjoint == False`)
        weights : ndarray of floats, shape (nchannels,)
            weights for the features/channels of x
        imagedims : tuple of ints, length ndims
            resolution of the image grid
        imageh : ndarray of floats, shape (ndims,)
            step sizes of the image grid
        adjoint : bool
            optionally apply the adjoint operator (reading from y and writing
            to x), i.e. `x[i,k] -= weights[k]*div[i,:,:]y[:,:,k]`.
        precond : bool
            optionally compute rowwise/colwise L1 norm of `diag(weights)D`.
    """
    npoints, ndims, nchannels = y.shape
    navgskips =  1 << (ndims - 1)
    avgskips = staggered_diff_avgskips(imagedims)
    skips = imagedim_skips(imagedims)
    coords = np.zeros(ndims, dtype=np.int64)

    for k in range(nchannels):
        for t in range(ndims):
            coords.fill(0)
            for i in range(npoints):
                in_range = True
                for dc in range(ndims-1,-1,-1):
                    if coords[dc] >= imagedims[dc] - 1:
                        in_range = False
                        break

                if in_range:
                    bk = weights[k]/(navgskips*imageh[t])

                    for avgskip in avgskips[t]:
                        base = i + avgskip
                        if precond:
                            if adjoint:
                                x[base + skips[t],k] += np.abs(bk)
                                x[base,k]            += np.abs(bk)
                            else:
                                y[i,t,k] += 2*np.abs(bk)
                        else:
                            if adjoint:
                                x[base + skips[t],k] += bk*y[i,t,k]
                                x[base,k]            -= bk*y[i,t,k]
                            else:
                                y[i,t,k] += bk*x[base + skips[t],k]
                                y[i,t,k] -= bk*x[base,k]

                for dd in range(ndims-1,-1,-1):
                    coords[dd] += 1
                    if coords[dd] >= imagedims[dd]:
                        coords[dd] = 0
                    else:
                        break

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

    def _call_cpu(self, x, y=None, add=False):
        assert y is not None
        x = self.x.vars(x)[0]
        y = self.y.vars(y)[0]
        if not add: y.fill(0.0)
        gradient(x, y, self.imagedims, self.imageh, self.weights)

    def rowwise_lp(self, y, p=1, add=False):
        assert(p == 1)
        y = self.y.vars(y)[0]
        x = np.empty(self.x[0]['shape'])
        if not add: y.fill(0.0)
        gradient(x, y, self.imagedims, self.imageh, self.weights, precond=True)

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

    def _call_cpu(self, x, y=None, add=False):
        assert y is not None
        x = self.x.vars(x)[0]
        y = self.y.vars(y)[0]
        if not add: y.fill(0.0)
        divergence(x, y, self.imagedims, self.imageh, self.weights)

    def rowwise_lp(self, y, p=1, add=False):
        assert(p == 1)
        y = self.y.vars(y)[0]
        x = np.empty(self.x[0]['shape'])
        if not add: y.fill(0.0)
        divergence(x, y, self.imagedims, self.imageh, self.weights, precond=True)

@numba.njit
def laplacian_neumann(x, y, imagedims, imageh, precond=False):
    """ Computes `y[i,k] += Delta[i,:]x[:,k]`.

    Discrete Laplacian with Neumann boundary conditions (self-adjoint)

    Args:
        x : ndarray of floats, shape (npoints, nchannels)
            function to be derived
        y : ndarray of floats, shape (npoints, nchannels)
            Laplacian of x
        imagedims : tuple of ints, length ndims
            resolution of the image grid
        imageh : ndarray of floats, shape (ndims,)
            step sizes of the image grid
        precond : bool
            optionally compute rowwise/colwise L1 norm of `Delta`
    """
    npoints, nchannels = y.shape
    ndims = len(imagedims)
    skips = imagedim_skips(imagedims)
    coords = np.zeros(ndims, dtype=np.int64)

    for k in range(nchannels):
        coords.fill(0)
        for i in range(npoints):
            for t in range(ndims):
                invh2 = 1.0/imageh[t]**2
                if coords[t] > 0:
                    if precond:
                        y[i,k] += 2*invh2
                    else:
                        y[i,k] += invh2*(x[i - skips[t],k] - x[i,k])

                if coords[t] < imagedims[t] - 1:
                    if precond:
                        y[i,k] += 2*invh2
                    else:
                        y[i,k] += invh2*(x[i + skips[t],k] - x[i,k])

            for dd in range(ndims-1,-1,-1):
                coords[dd] += 1
                if coords[dd] >= imagedims[dd]:
                    coords[dd] = 0
                else:
                    break

@numba.njit
def laplacian_curvature(x, y, imagedims, imageh, adjoint=False, precond=False):
    """ Computes `y[i,k] += Delta[i,:]x[:,k]`.

    Discrete Laplacian with curvature boundary conditions (the Laplacian is
    vanishing at the boundary), not self-adjoint.

    Args:
        x : ndarray of floats, shape (npoints, nchannels)
            function to be derived
        y : ndarray of floats, shape (npoints, nchannels)
            (adjoint) Laplacian of x
        imagedims : tuple of ints, length ndims
            resolution of the image grid
        imageh : ndarray of floats, shape (ndims,)
            step sizes of the image grid
        adjoint : bool
            optionally compute adjoint operator
        precond : bool
            optionally compute rowwise/colwise L1 norm of `Delta`
    """
    npoints, nchannels = y.shape
    ndims = len(imagedims)
    skips = imagedim_skips(imagedims)
    coords = np.zeros(ndims, dtype=np.int64)

    for k in range(nchannels):
        coords.fill(0)
        for i in range(npoints):
            in_range = True
            for dc in range(ndims-1,-1,-1):
                if coords[dc] == 0 or coords[dc] == imagedims[dc]-1:
                    in_range = False
                    break

            if in_range:
                for t in range(ndims):
                    invh2 = 1.0/imageh[t]**2
                    if adjoint:
                        if precond:
                            y[i - skips[t],k] += invh2
                            y[i + skips[t],k] += invh2
                            y[i,k]            += 2*invh2
                        else:
                            y[i - skips[t],k] += invh2*x[i,k]
                            y[i + skips[t],k] += invh2*x[i,k]
                            y[i,k]            -= 2*invh2*x[i,k]
                    else:
                        if precond:
                            y[i,k] += 4*invh2
                        else:
                            y[i,k] += invh2*x[i - skips[t],k]
                            y[i,k] += invh2*x[i + skips[t],k]
                            y[i,k] -= 2*invh2*x[i,k]

            for dd in range(ndims-1,-1,-1):
                coords[dd] += 1
                if coords[dd] >= imagedims[dd]:
                    coords[dd] = 0
                else:
                    break

class LaplacianOp(LinOp):
    """ Laplacian operator with different boundary conditions """
    def __init__(self, imagedims, nchannels, imageh=None,
                       boundary="neumann", adjoint=None):
        LinOp.__init__(self)
        npoints = np.prod(imagedims)
        ndims = len(imagedims)
        self.imagedims = imagedims
        self.imageh = np.ones(ndims) if imageh is None else imageh
        self.boundary = boundary
        self.nchannels = nchannels
        self.x = Variable((npoints, nchannels))
        self.y = Variable((npoints, nchannels))
        self._kernel = None
        if self.boundary == "neumann":
            self.adjoint = self
        elif self.boundary[:9] == "curvature":
            if adjoint is None:
                adj_boundary = "curvature_adj"
                if self.boundary == adj_boundary:
                    adj_boundary = "curvature"
                self.adjoint = LaplacianOp(imagedims, nchannels, imageh=imageh,
                                           boundary=adj_boundary, adjoint=self)
            else:
                self.adjoint = adjoint
        else:
            raise Exception("Unknown boundary conditions: %s" % self.boundary)

    def prepare_gpu(self, type_t="double"):
        if self._kernel is not None: return
        npoints = np.prod(self.imagedims)
        ndims = len(self.imagedims)
        skips = imagedim_skips(self.imagedims)
        constvars = {
            'LAPLACIAN': 1,
            'ADJOINT': 1 if self.boundary[-3:] == "adj" else 0,
            'boundary_conditions': self.boundary[0].upper(),
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

    def _call_cpu(self, x, y=None, add=False):
        assert y is not None
        x = self.x.vars(x)[0]
        y = self.y.vars(y)[0]
        if not add: y.fill(0.0)
        if self.boundary == "neumann":
            laplacian_neumann(x, y, self.imagedims, self.imageh)
        else:
            adj = self.boundary[-3:] == "adj"
            laplacian_curvature(x, y, self.imagedims, self.imageh, adjoint=adj)

    def rowwise_lp(self, y, p=1, add=False):
        assert(p == 1)
        y = self.y.vars(y)[0]
        x = np.empty(self.x[0]['shape'])
        if not add: y.fill(0.0)
        if self.boundary == "neumann":
            laplacian_neumann(x, y, self.imagedims, self.imageh, precond=True)
        else:
            adj = self.boundary[-3:] == "adj"
            laplacian_curvature(x, y, self.imagedims, self.imageh,
                                adjoint=adj, precond=True)
