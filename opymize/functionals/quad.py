
from opymize import Variable, Functional
from opymize.operators.affine import ShiftScaleOp
from opymize.operators.proj import QuadEpiProj

import numpy as np

class SSD(Functional):
    """ 0.5*<x-f, x-f>_b + shift
        where b is the volume element
    """
    def __init__(self, data, vol=None, shift=0, mask=None, conj=None):
        Functional.__init__(self)
        self.x = Variable(data.shape)
        self.f = np.atleast_2d(data)
        self.shift = shift
        self.vol = np.ones(data.shape[1]) if vol is None else vol
        self.mask = np.ones(data.shape[0], dtype=bool) if mask is None else mask
        if conj is None:
            cj_vol = 1.0/self.vol
            cj_data = np.zeros_like(self.f)
            cj_data[self.mask,:] = np.einsum('ik,k->ik', self.f[self.mask,:], -self.vol)
            cj_shift = -0.5*np.einsum('ik,k->', cj_data**2, cj_vol)
            cj_shift -= self.shift
            self.conj = SSD(cj_data, shift=cj_shift, vol=cj_vol, mask=mask, conj=self)
        else:
            self.conj = conj
        prox_shift = np.zeros_like(self.f)
        prox_shift[self.mask,:] = self.f[self.mask,:]
        prox_shift = prox_shift.ravel()
        self._prox = ShiftScaleOp(self.x.size, prox_shift, 0.5, 1.0)

    def __call__(self, x, grad=False):
        x = self.x.vars(x)[0]
        val = 0.5*np.einsum('ik,k->', (x - self.f)[self.mask,:]**2, self.vol)
        val += self.shift
        infeas = 0
        result = (val, infeas)
        if grad:
            df = np.zeros_like(x)
            df[self.mask,:] = np.einsum('ik,k->ik', (x - self.f)[self.mask,:], self.vol)
            return result, df.ravel()
        else:
            return result

    def prox(self, tau):
        msk = self.mask
        tauvol = np.zeros_like(self.f)
        tauvol[msk,:] = (tau*np.ones(self.f.size)).reshape(self.f.shape)[msk,:]
        tauvol[msk,:] = np.einsum('ik,k->ik', tauvol[msk,:], self.vol)
        self._prox.a = 1.0/(1.0 + tauvol.ravel())
        self._prox.b = tauvol.ravel()
        if hasattr(self._prox, 'gpuvars'):
            self._prox.gpuvars['a'][:] = self._prox.a
            self._prox.gpuvars['b'][:] = self._prox.b
        return self._prox

def quad_dual_coefficients(a, b, c):
    # f_i(x) := 0.5*a*|x|^2 + <b[i],x> + c[i]
    # f_i^*(x) = 0.5/a*|x - b[i]|^2 - c[i]
    #          = 0.5/a*|x|^2 + <-b[i]/a,x> + (0.5/a*|b[i]|^2 - c[i])
    return (1.0/a, -b/a, 0.5/a*(b**2).sum(axis=-1) - c)

class QuadSupport(Functional):
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
            self.conj = QuadSupport(N, M, a=da, b=db, c=dc, conj=self)
        else:
            self.conj = conj
        self._prox = QuadEpiProj(N, M, a=a, b=b, c=c)

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
