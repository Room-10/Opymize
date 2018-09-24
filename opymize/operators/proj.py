
from opymize import Variable, Operator, LinOp
from opymize.linear import NihilOp

import numpy as np
import numba

try:
    import opymize.tools.gpu
    from opymize.tools.gpu import prepare_kernels
    from pkg_resources import resource_stream
    from pycuda.elementwise import ElementwiseKernel
except:
    # no cuda support?
    pass

class PosProj(Operator):
    """ T(x) = max(0, x), in the elementwise sense"""
    def __init__(self, N):
        Operator.__init__(self)
        self.x = Variable(N)
        self.y = Variable(N)
        self._jacobian = NihilOp(N, keep=self.x.new(dtype=bool))

    def prepare_gpu(self, type_t="double"):
        self._kernel = ElementwiseKernel("%s *x, %s *y" % ((type_t,)*2),
            "y[i] = (x[i] < 0) ? 0 : x[i]")
        self._kernel_add = ElementwiseKernel("%s *x, %s *y" % ((type_t,)*2),
            "y[i] += (x[i] < 0) ? 0 : x[i]")

    def _call_gpu(self, x, y=None, add=False, jacobian=False):
        y = x if y is None else y
        if add: self._kernel_add(x, y)
        else: self._kernel(x, y)

        if jacobian:
            self._jacobian.keep = (x.get() > 0)
            return self._jacobian

    def _call_cpu(self, x, y=None, add=False, jacobian=False):
        y = x if y is None else y
        if jacobian:
            self._jacobian.keep = (x > 0)
        if add:
            y += np.fmax(0, x)
        else:
            np.fmax(0, x, out=y)
        if jacobian: return self._jacobian

class NegProj(Operator):
    """ T(x) = min(0, x), in the elementwise sense"""
    def __init__(self, N):
        Operator.__init__(self)
        self.x = Variable(N)
        self.y = Variable(N)
        self._jacobian = NihilOp(N, keep=self.x.new(dtype=bool))

    def prepare_gpu(self, type_t="double"):
        self._kernel = ElementwiseKernel("%s *x, %s *y" % ((type_t,)*2),
            "y[i] = (x[i] > 0) ? 0 : x[i]")
        self._kernel_add = ElementwiseKernel("%s *x, %s *y" % ((type_t,)*2),
            "y[i] += (x[i] > 0) ? 0 : x[i]")

    def _call_gpu(self, x, y=None, add=False, jacobian=False):
        y = x if y is None else y
        if add:
            self._kernel_add(x, y)
        else:
            self._kernel(x, y)

        if jacobian:
            self._jacobian.keep = (x.get() < 0)
            return self._jacobian

    def _call_cpu(self, x, y=None, add=False, jacobian=False):
        y = x if y is None else y
        if jacobian:
            self._jacobian.keep = (x < 0)
        if add:
            y += np.fmin(0, x)
        else:
            np.fmin(0, x, out=y)
        if jacobian: return self._jacobian

def norm_projection(x, xnorms, lbd, matrixnorm="spectral"):
    """ Project each x^i to the norm ball of radius lbd

    Args:
        x : numpy array of shape (N, M1, M2) where `M1 in (1,2) or M2 in (1,2)`
        xnorms : numpy array of shape (N,)
        lbd : radius of the norm ball
        matrixnorm : one of 'spectral' or 'frobenius'
    Returns:
        nothing, the result is stored in place!
        the reciprocal norms are stored in `xnorms`
    """
    N, M1, M2 = x.shape

    if M1 == 1 or M2 == 1 or matrixnorm == "frobenius":
        np.einsum('ilm,ilm->i', x, x, out=xnorms)
        np.sqrt(xnorms, out=xnorms)
        extind = (xnorms > lbd)
        np.reciprocal(xnorms, out=xnorms, where=extind)
        x[extind,:,:] *= xnorms[extind,None,None]
        x[extind,:,:] *= lbd
        return extind
    elif M1 == 2 or M2 == 2 or matrixnorm == "spectral":
        spectral_projection_2d(x, lbd)
    else:
        raise Exception("Dimension error: M1={:d}, M2={:d}".format(M1, M2))

@numba.jit
def spectral_projection_2d(x, lbd):
    """ Project the matrices x^i to the spectral ball of radius lbd

    Args:
        x : numpy array of shape (N, M1, M2) where `M1 in (1,2) or M2 in (1,2)`
            The x^i are projected to the ball with radius lbd.
        lbd : radius of the spectral ball
    Returns:
        nothing, the result is stored in place!
    """
    N, M1, M2 = x.shape
    V = np.empty((2,2))
    C = np.empty((2,2))
    S = np.zeros((2,2))
    for i in range(N):
        if M2 == 2:
            A = x[i]
        else:
            A = x[i].T
        np.dot(A.T, A, out=C)

        # Compute eigenvalues
        trace = C[0,0] + C[1,1]
        d = C[0,0]*C[1,1] - C[0,1]*C[0,1]
        d = np.sqrt(max(0.0, 0.25*trace**2 - d))
        lmin, lmax = max(0.0, 0.5*trace - d), max(0.0, 0.5*trace + d)
        smin, smax = np.sqrt(lmin), np.sqrt(lmax)

        if smax > lbd:
            # Compute orthonormal eigenvectors
            if C[0,1] == 0.0:
                if C[0,0] >= C[1,1]:
                    V[0,1] = V[1,0] = 0.0
                    V[0,0] = V[1,1] = 1.0
                else:
                    V[0,1] = V[1,0] = 1.0
                    V[0,0] = V[1,1] = 0.0
            else:
                V[0,0] = V[0,1] = C[0,1]
                V[1,0] = lmax - C[0,0]
                V[1,1] = lmin - C[0,0]
                Vnorm = np.sqrt(V[0,0]**2 + V[1,0]**2)
                V[0,0] /= Vnorm
                V[1,0] /= Vnorm
                Vnorm = np.sqrt(V[0,1]**2 + V[1,1]**2)
                V[0,1] /= Vnorm
                V[1,1] /= Vnorm

            # Thresholding of eigenvalues
            S[0,0] = min(smax, lbd)/smax
            S[1,1] = min(smin, lbd)/smin if smin > 0.0 else 0.0

            # proj(A) = A * V * S * V^T
            A[:] = np.dot(A, V.dot(S).dot(V.T))

class L1NormsProj(Operator):
    """ T(x) = proj[lbd](x)
    Supported norms are 'frobenius' and 'spectral'
    """
    def __init__(self, N, M, lbd, matrixnorm):
        Operator.__init__(self)
        assert matrixnorm in ['frobenius', 'spectral']
        self.M = M
        self.N = N
        self.x = Variable((N,) + M)
        self.y = self.x
        self.lbd = lbd
        self._xnorms = np.zeros((N,), order='C')
        self.matrixnorm = matrixnorm
        if matrixnorm == "frobenius":
            self._jacobian = L12ProjJacobian(self.N, self.M, self.lbd)

    def prepare_gpu(self, type_t="double"):
        constvars = {
            'lbd': self.lbd,
            'N': self.N, 'M1': self.M[0], 'M2': self.M[1],
            'matrixnorm': self.matrixnorm[0].upper(),
            'TYPE_T': type_t
        }
        fd = 3 if type_t == "double" else 4
        files = [resource_stream('opymize.operators', 'proj.cu')]
        templates = [("l1normsproj", "P", (self.N, 1, 1), (fd*256, 1, 1))]
        self._kernel = prepare_kernels(files, templates, constvars)['l1normsproj']

    def _call_gpu(self, x, y=None, add=False, jacobian=False):
        assert not add
        assert not jacobian
        if y is not None:
            y[:] = x.copy()
            x = y
        self._kernel(x)

    def _call_cpu(self, x, y=None, add=False, jacobian=False):
        assert not add
        assert self.matrixnorm == "frobenius" or not jacobian
        x = self.x.vars(x)[0]
        if y is not None:
            y = self.y.vars(y)[0]
            y[:] = x
        else:
            y = x
            if jacobian:
                xn = self._jacobian.xbar_normed
                xn[:] = x.reshape(xn.shape)
                x = xn.reshape(x.shape)
        extind = norm_projection(y, self._xnorms, self.lbd,
                                 matrixnorm=self.matrixnorm)
        if jacobian:
            self._jacobian.update(x, extind, self._xnorms)
            return self._jacobian

class L12ProjJacobian(LinOp):
    """ Jacobian of L1NormsProj for Frobenius norm """
    def __init__(self, N, M, lbd):
        # xnorms[i] = 1.0/|xbar[i,:,:]|_2
        #  exterior = (xbar > lbd)
        LinOp.__init__(self)
        self.x = Variable((N, M[0]*M[1]))
        self.y = self.x
        self.lbd = lbd
        self.adjoint = self
        self.extind = np.zeros(N, dtype=bool)
        self.intind = np.zeros(N, dtype=bool)
        self.xbar_normed = self.x.vars(self.x.new())[0]
        self.lbd_norms = np.zeros(N)

    def update(self, xbar, exterior, xnorms):
        self.extind[:] = exterior
        self.intind[:] = np.logical_not(exterior)

        self.xbar_normed[:] = xbar.reshape(self.xbar_normed.shape)
        self.xbar_normed[exterior,:] *= xnorms[exterior,None]

        self.lbd_norms[:] = self.lbd*xnorms

    def _call_cpu(self, x, y=None, add=False):
        x = self.x.vars(x)[0]
        if add or y is None:
            yy = self.y.vars(self.y.new())[0]
        else:
            yy = self.y.vars(y)[0]

        # xn[i,k] = xbar[i,k]/|xbar[i,:]|
        xn = self.xbar_normed

        # y[norms <= lbd] = x
        yy[self.intind,:] = x[self.intind,:]

        # y[norms > lbd] = lbd/|xbar|*(x - <xn,x>*xn)
        yy[self.extind,:] = xn[self.extind,:]
        yy[self.extind,:] *= -np.einsum('ik,ik->i',
            xn[self.extind,:], x[self.extind,:])[:,None]
        yy[self.extind,:] += x[self.extind,:]
        yy[self.extind,:] *= self.lbd_norms[self.extind,None]

        if y is None:
            if add: x += yy
            else: x[:] = yy
        elif add:
            y += yy
