
from opymize import Variable, Operator, LinOp
from opymize.linear import NihilOp

import numpy as np
import numba

import cvxopt
import cvxopt.solvers
cvxopt.solvers.options['show_progress'] = False

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
    V = np.empty((2,2), dtype=x.dtype)
    C = np.empty((2,2), dtype=x.dtype)
    S = np.zeros((2,2), dtype=x.dtype)
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
            'L1_NORMS_PROJ': 1,
            'lbd': self.lbd,
            'N': self.N, 'M1': self.M[0], 'M2': self.M[1],
            'matrixnorm': self.matrixnorm[0].upper(),
            'TYPE_T': type_t,
        }
        for f in ['fmin','fmax','sqrt','hypot']:
            constvars[f.upper()] = f if type_t == "double" else (f+"f")
        files = [resource_stream('opymize.operators', 'proj.cu')]
        templates = [("l1normsproj", "P", (self.N, 1, 1), (640, 1, 1))]
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

class EpigraphProj(Operator):
    """ T(x)[j,i] = proj[epi(f[i]_j*)](x[j,i])

    Project onto the epigraphs epi(f[i]_j*) of the convex conjugates of convex
    pieces f[i]_j of functions f[i].
    The functions f[i] are piecewise linear with common base points
    {v[k] for k in I[i]} and values b[i,k] = f[i](v[k]) for k in I[i].
    By f[i]_j we denote the restriction of f[i] to a subset (described by J[j])
    such that f[i]_j is convex.

        f[i]_j*(x) = max_{k : I[i,J[j]][k] == True} <v[k],x> - b[i,k]

    In other words, y = T(x)[j,i] is the solution to the inequality constrained
    quadratic program

        minimize 0.5*||y - x[j,i]||**2
        s.t. <v[k],y[:-1]> - y[-1] <= b[i,k] for any k with I[i,J[j]][k] == True

    The solution is computed using a QP solver.
    """
    def __init__(self, I, J, v, b):
        """
        Args:
            I : ndarray of bools, shape (nfuns, npoints)
            J : ndarray of ints, shape (nregions, nsubpoints)
            v : ndarray of floats, shape (npoints, 2)
            b : ndarray of floats, shape (nfuns, npoints)
        """
        Operator.__init__(self)

        nfuns, npoints = I.shape
        nregions, nsubpoints = J.shape

        self.x = Variable((nregions, nfuns, 3))
        self.y = self.x

        self.I = I
        self.J = J
        self.v = v
        self.b = b

    def prepare_gpu(self, type_t="double"):
        from pycuda import gpuarray

        nfuns, npoints = self.I.shape
        nregions, nsubpoints = self.J.shape

        counts = np.zeros((nfuns*nregions,), dtype=np.int32)
        counts[:] = [np.count_nonzero(Ii[Jj]) for Ii in self.I for Jj in self.J]

        indices = np.zeros((nfuns*nregions,), dtype=np.int64)
        np.cumsum(counts[:-1], out=indices[1:])
        total_count = indices[-1] + counts[-1]

        counts = counts.reshape((nfuns, nregions))
        indices = indices.reshape((nfuns, nregions))

        np_dtype = np.float64 if type_t == "double" else np.float32
        A_gpu = np.zeros((total_count,2), dtype=np_dtype)
        b_gpu = np.zeros((total_count,), dtype=np_dtype)
        for i in range(nfuns):
            for j in range(nregions):
                mask, idx, count = self.I[i,self.J[j]], indices[i,j], counts[i,j]
                A_gpu[idx:idx+count,:] = self.v[self.J[j]][mask]
                b_gpu[idx:idx+count] = self.b[i,self.J[j]][mask]

        constvars = {
            'EPIGRAPH_PROJ': 1,
            'nfuns': nfuns, 'nregions': nregions,
            'counts': counts, 'indices': indices,
            'A_STORE': A_gpu, 'B_STORE': b_gpu,
            'term_maxiter': 2500, 'term_tolerance': 1e-9,
            'TYPE_T': type_t,
        }
        for f in ['fabs']:
            constvars[f.upper()] = f if type_t == "double" else (f+"f")
        files = [resource_stream('opymize.operators', 'proj.cu')]
        templates = [("epigraphproj", "P", (nregions, nfuns, 1), (24, 12, 1))]
        self._kernel = prepare_kernels(files, templates, constvars)['epigraphproj']

    def _call_gpu(self, x, y=None, add=False, jacobian=False):
        assert not add
        assert not jacobian
        if y is not None:
            y[:] = x.copy()
            x = y
        self._kernel(x)

    def _call_cpu(self, x, y=None, add=False, jacobian=False):
        assert not add
        assert not jacobian
        x = self.x.vars(x)[0]
        for i in range(self.I.shape[0]):
            for j in range(self.J.shape[0]):
                xji = x[j,i]
                mask = self.I[i,self.J[j]]
                b = self.b[i,self.J[j]][mask]
                A = np.zeros((b.size,3))
                A[:,0:-1] = self.v[self.J[j]][mask]
                A[:,-1] = -1.0

                #   Now solve
                # minimize  0.5*||y - xji||**2  s.t.  A y <= b
                #   or
                # minimize  0.5*||y||**2 - <xji,y>  s.t.  A y <= b
                #   which is equivalent to
                # minimize 0.5*||A' z||**2 - <A xji - b,z> s.t. z >= 0
                #   or
                # minimize 0.5*||A' z - xji||**2 + <b,z> s.t. z >= 0
                #   for y = xji - A' z.

                P = cvxopt.spmatrix(1.0, range(b.size), range(b.size))
                q = -cvxopt.matrix(xji)
                G = cvxopt.matrix(A)
                h = cvxopt.matrix(b)
                xji[:] = np.array(cvxopt.solvers.qp(P, q, G, h)['x']).ravel()
