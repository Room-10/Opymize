
from opymize import Variable, Functional
from opymize.operators.proj import EpigraphProj, epigraph_Ab, QuadEpiProj

import numpy as np

class EpigraphInd(Functional):
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
            self.conj = EpigraphSupp(I, J, v, b, conj=self)
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
EpigraphFct = EpigraphInd

class EpigraphSupp(Functional):
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
            self.conj = EpigraphInd(I, J, v, b, conj=self)
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
EpigraphSupportFct = EpigraphSupp

def quad_dual_coefficients(a, b, c):
    # f_i(x) := 0.5*a*|x|^2 + <b[i],x> + c[i]
    # f_i^*(x) = 0.5/a*|x - b[i]|^2 - c[i]
    #          = 0.5/a*|x|^2 + <-b[i]/a,x> + (0.5/a*|b[i]|^2 - c[i])
    return (1.0/a, -b/a, 0.5/a*(b**2).sum(axis=-1) - c)

class QuadEpiSupp(Functional):
    """ \sum_i -x[i,-1]*f_i(-x[i,:-1]/x[i,-1]) if x[i,-1] < 0
        and inf if x[i,-1] >= 0

        f_i(x) := 0.5*a*|x|^2 + <b[i],x> + c[i]
    """
    def __init__(self, N, M, a=1.0, b=None, c=None, conj=None):
        Functional.__init__(self)
        assert a > 0
        self.x = Variable((N, M + 1))
        self.a = a
        self.b = np.zeros((N, M)) if b is None else b
        self.c = np.zeros((N,)) if c is None else c
        if conj is None:
            da, db, dc = quad_dual_coefficients(self.a, self.b, self.c)
            self.conj = QuadEpiInd(N, M, a=da, b=db, c=dc, conj=self)
        else:
            self.conj = conj

    def __call__(self, x, grad=False):
        assert not grad
        a, b, c = self.a, self.b, self.c
        x = self.x.vars(x)[0]
        msk = x[:,-1] < -1e-8
        x1, x2 = x[msk,:-1], -x[msk,-1]
        val = (0.5*a*x1**2/x2[:,None] + b[msk]*x1).sum(axis=1) + x2*c[msk]
        val = val.sum()
        if np.all(msk):
            infeas = 0
        else:
            infeas = np.linalg.norm(x[np.logical_not(msk),-1], ord=np.inf)
        return (val, infeas)
QuadSupport = QuadEpiSupp

class QuadEpiInd(Functional):
    """ \sum_i \delta_{f_i(x[i,:-1]) \leq x[i,-1]}
        f_i(x) := 0.5*a*|x|^2 + <b[i],x> + c[i]
     """
    def __init__(self, N, M, a=1.0, b=None, c=None, conj=None):
        Functional.__init__(self)
        assert a > 0
        self.x = Variable((N, M + 1))
        self.a = a
        self.b = np.zeros((N, M)) if b is None else b
        self.c = np.zeros((N,)) if c is None else c
        if conj is None:
            da, db, dc = quad_dual_coefficients(self.a, self.b, self.c)
            self.conj = QuadEpiSupp(N, M, a=da, b=db, c=dc, conj=self)
        else:
            self.conj = conj
        self._prox = QuadEpiProj(N, M, alph=a, b=b, c=c)

    def __call__(self, x, grad=False):
        assert not grad
        x = self.x.vars(x)[0]
        fx = (0.5*self.a*x[:,:-1]**2 + self.b*x[:,:-1]).sum(axis=1) + self.c
        dif = fx - x[:,-1]
        val = 0
        infeas = np.linalg.norm(np.fmax(0, dif), ord=np.inf)
        result = (val, infeas)
        if grad:
            result = result, self.x.new()
        return result

    def prox(self, tau):
        # independent of tau!
        return self._prox

class HuberPerspective(Functional):
    """ \sum_i -x[i,-1]*f(-x[i,:-1]/x[i,-1]) if x[i,-1] < 0
        and inf if x[i,-1] >= 0

        f(x) := |  lbd*(0.5/alph*|x|^2),   if |x| < alph,
                |  lbd*(|x| - alph/2),     if |x| > alph.
    """
    def __init__(self, N, M, lbd=1.0, alph=1.0, conj=None):
        Functional.__init__(self)
        assert lbd > 0
        assert alph > 0
        self.x = Variable((N, M + 1))
        self.lbd = lbd
        self.alph = alph
        if conj is None:
            dlbd, dalph = self.lbd, self.alph/self.lbd
            self.conj = TruncQuadEpiInd(N, M, lbd=dlbd, alph=dalph, conj=self)
        else:
            self.conj = conj

    def __call__(self, x, grad=False):
        assert not grad
        lbd, alph = self.lbd, self.alph
        x = self.x.vars(x)[0]
        msk = x[:,-1] < -1e-8
        x1, x2 = x[msk,:-1], -x[msk,-1]
        xnorm = np.linalg.norm(x1/x2[:,None], axis=-1)
        qmsk = xnorm <= alpha
        qmsk_n = np.logical_not(qmsk)
        val = (x2[qmsk]*lbd*0.5/alph*xnorm[qmsk]**2).sum()
        val += (x2[qmsk_n]*lbd*(xnorm[qmsk_n] - alph/2).sum()
        if np.all(msk):
            infeas = 0
        else:
            infeas = np.linalg.norm(x[np.logical_not(msk),-1], ord=np.inf)
        return (val, infeas)

class TruncQuadEpiInd(Functional):
    """ \sum_i \delta_{|x| \leq lbd} + \delta_{f(x[i,:-1]) \leq x[i,-1]}
        f(x) := 0.5*alph*|x|^2
     """
    def __init__(self, N, M, lbd=1.0, alph=1.0, conj=None):
        Functional.__init__(self)
        assert lbd > 0
        assert alph > 0
        self.x = Variable((N, M + 1))
        self.lbd = lbd
        self.alph = alph
        if conj is None:
            dlbd, dalph = self.lbd, self.alph*self.lbd
            self.conj = HuberPerspective(N, M, lbd=dlbd, alph=dalph, conj=self)
        else:
            self.conj = conj
        self._prox = QuadEpiProj(N, M, lbd=self.lbd, alph=self.alph)

    def __call__(self, x, grad=False):
        assert not grad
        x = self.x.vars(x)[0]
        val = 0
        x1norm = np.linalg.norm(x[:,:-1], axis=-1)
        dif = x1norm - self.lbd
        infeas = np.linalg.norm(np.fmax(0, dif), ord=np.inf)
        fx = 0.5*alph*np.fmin(self.lbd, x1norm)**2
        dif = fx - x[:,-1]
        infeas += np.linalg.norm(np.fmax(0, dif), ord=np.inf)
        result = (val, infeas)
        if grad:
            result = result, self.x.new()
        return result

    def prox(self, tau):
        # independent of tau!
        return self._prox
