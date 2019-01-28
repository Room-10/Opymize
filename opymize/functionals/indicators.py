
from opymize import Variable, Functional
from opymize.operators import ConstOp, ConstrainOp, PosProj, NegProj, \
                              EpigraphProj, epigraph_Ab

import numpy as np
from numpy.linalg import norm

class IndicatorFct(Functional):
    """ F(x) = c2 if x==c1 else infty (use broadcasting in c1 if necessary) """
    def __init__(self, N, c1=0, c2=0, conj=None):
        Functional.__init__(self)
        self.c1 = c1
        self.c2 = c2
        self.x = Variable(N)
        from opymize.functionals import AffineFct
        self.conj = AffineFct(N, b=-c2, c=c1, conj=self) if conj is None else conj
        self._prox = ConstOp(N, self.x.new() + c1)

    def __call__(self, x, grad=False):
        val = self.c2
        infeas = norm(x - self.c1, ord=np.inf)
        result = (val, infeas)
        if grad:
            result = result, self.x.new()
        return result

    def prox(self, tau):
        # independent of tau
        return self._prox

class ConstrainFct(Functional):
    """ F(x) = 0 if x[mask,:]==c[mask,:] else infty
        The mask is only applied to the first component of x
    """
    def __init__(self, mask, c, conj=None):
        Functional.__init__(self)
        self.x = Variable(c.shape)
        self.mask = mask
        self.c = c
        if conj is None:
            from opymize.functionals import MaskedAffineFct
            self.conj = MaskedAffineFct(mask, c, conj=self)
        else:
            self.conj = conj
        self._prox = ConstrainOp(mask, c)

    def __call__(self, x, grad=False):
        x = self.x.vars(x)[0]
        val = 0
        infeas = norm(x[self.mask,:] - self.c[self.mask,:], ord=np.inf)
        result = (val, infeas)
        if grad:
            result = result, self.x.new()
        return result

    def prox(self, tau):
        # independent of tau
        return self._prox

class PositivityFct(Functional):
    """ F(x) = 0 if x >= 0 else infty """
    def __init__(self, N, conj=None):
        Functional.__init__(self)
        self.x = Variable(N)
        self.conj = NegativityFct(N, conj=self) if conj is None else conj
        self._prox = PosProj(N)

    def __call__(self, x, grad=False):
        val = 0
        infeas = norm(np.fmin(0, x), ord=np.inf)
        result = (val, infeas)
        if grad:
            result = result, self.x.new()
        return result

    def prox(self, tau):
        # independent of tau
        return self._prox

class NegativityFct(Functional):
    """ F(x) = 0 if x <= 0 else infty """
    def __init__(self, N, conj=None):
        Functional.__init__(self)
        self.x = Variable(N)
        self.conj = PositivityFct(N, conj=self) if conj is None else conj
        self._prox = NegProj(N)

    def __call__(self, x, grad=False):
        val = 0
        infeas = norm(np.fmax(0, x), ord=np.inf)
        result = (val, infeas)
        if grad:
            result = result, self.x.new()
        return result

    def prox(self, tau):
        # independent of tau
        return self._prox

class EpigraphFct(Functional):
    """ F(x) = 0 if x[j,i] \in epi(f[i]_j*) for every j,i else infty

    More precisely:

        F(x) = 0 if <v[k],x[j,i,:-1]> - b[i,k] <= x[j,i,-1]
                    for any i,j,k with I[i,J[j]][k] == True
    """
    def __init__(self, I, J, v, b, conj=None):
        """
        Args:
            I : ndarray of bools, shape (nfuns, npoints)
            J : ndarray of ints, shape (nregions, nsubpoints)
            v : ndarray of floats, shape (npoints, ndim)
            b : ndarray of floats, shape (nfuns, npoints)
        """
        Functional.__init__(self)

        nfuns, npoints = I.shape
        nregions, nsubpoints = J.shape
        ndim = v.shape[1]
        self.I, self.J, self.v, self.b = I, J, v, b

        self.x = Variable((nregions, nfuns, ndim+1))

        if conj is None:
            self.conj = EpigraphSupportFct(I, J, v, b, conj=self)
        else:
            self.conj = conj

        self.A = self.conj.A
        self.b = self.conj.b
        self._prox = EpigraphProj(I, J, v, b, Ab=(self.A, self.b))

    def __call__(self, x, grad=False):
        val = 0
        infeas = max(0, np.amax(self.A.dot(x) - self.b))
        result = (val, infeas)
        if grad:
            result = result, self.x.new()
        return result

    def prox(self, tau):
        # independent of tau
        return self._prox

class EpigraphSupportFct(Functional):
    """ F(x) = max sum_ji <y,x[j,i]>  s.t. y \in epi(f[i]_j*)

    f[i] : piecewise linear function
    f[i]_j : restriction of f[i] to the simplicial region j
        The pieces f[i]_j are assumed to be convex.

    More precisely:

        F(x) = max sum_ji <y,x[j,i]>
               s.t. <v[k],y[:-1]> - b[i,k] <= y[-1]
                    for any i,j,k with I[i,J[j]][k] == True
    """
    def __init__(self, I, If, J, v, b, conj=None):
        """
        Args:
            I : ndarray of bools, shape (nfuns, npoints)
                Selection of support points used by each of the f[i].
            If : nfuns lists of nregions arrays, shape (nfaces,ndim+1) each
                Indices of faces forming the graph of each of the f[i]_j.
            J : ndarray of ints, shape (nregions, nsubpoints)
                Description of the simplicial regions.
            v : ndarray of floats, shape (npoints, ndim)
                All possible support points of the f[i].
            b : ndarray of floats, shape (nfuns, npoints)
                Values of the f[i] at the support points. Only values at the
                support points selected by I are used.
        """
        Functional.__init__(self)

        nfuns, npoints = I.shape
        nregions, nsubpoints = J.shape
        ndim = v.shape[1]
        self.x = Variable((nregions, nfuns, ndim+1))
        self.A, self.b = epigraph_Ab(I, J, v, b)

        # Each f[i] is known by its values on support points v.
        # The following code computes the equations of the affine functions
        # that describe each f[i]_j.
        self.eqns = []
        for j in range(nregions):
            for i in range(nfuns):
                faces = If[i][j]
                points = v[J[j]][faces]
                vals = b[i,J[j]][faces]
                ptmats = points[:,1:] - points[:,:1]
                valbs = vals[:,1:] - vals[:,:1]
                Ab = np.zeros((faces.shape[0],ndim+1))
                Ab[:,:-1] = np.linalg.solve(ptmats, valbs)
                Ab[:,-1] = vals[:,0] - (Ab[:,:-1]*points[:,0,:]).sum(axis=-1)
                self.eqns.append(Ab)
        self.checks = -np.ones((nregions, ndim+1, ndim+1))
        self.checks[:,:,0:-1] = v[J[:,0:ndim+1],:]
        self.checks[:] = np.linalg.inv(self.checks).transpose(0,2,1)

        if conj is None:
            self.conj = EpigraphFct(I, J, v, b, conj=self)
        else:
            self.conj = conj

    def __call__(self, x, grad=False):
        assert not grad
        x = self.x.vars(x)[0]
        ndim_1 = x.shape[-1]
        infeas = np.einsum("jkl,jil->jik", self.checks, x).reshape(-1, ndim_1)
        infeas = np.amax(np.fmax(0, -infeas), axis=-1)
        xt = np.fmax(np.spacing(1), np.abs(x[:,:,-1]).ravel())
        x = x.copy().reshape(-1,ndim_1)
        x[:,-1] = 1.0
        x[:,:-1] /= xt[:,None]
        val = [Ab.dot(xij).max() for Ab,xij in zip(self.eqns, x)]
        return (xt*val).sum(), infeas.max()
