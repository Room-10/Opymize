
from opymize import Variable, Functional
from opymize.operators import ShiftScaleOp
from opymize.linear import IdentityOp

import numpy as np
from numpy.linalg import norm

class AffineFct(Functional):
    """ F(x) = b + sum_i c_i*x_i (use broadcasting in c if necessary) """
    def __init__(self, N, b=0, c=1, conj=None):
        Functional.__init__(self)
        self.b = b
        self.c = c
        self.x = Variable(N)
        from opymize.functionals import IndicatorFct
        self.conj = IndicatorFct(N, c1=c, c2=b, conj=self) if conj is None else conj
        self._prox = ShiftScaleOp(N, self.c, 1, -1)

    def __call__(self, x, grad=False):
        infeas = 0 # smooth

        val = self.b
        if type(self.c) is np.ndarray:
            val += np.einsum('i,i->', x, self.c)
        elif self.c != 0:
            val += self.c*np.einsum('i->',x)

        result = (val, infeas)
        if grad:
            if type(self.c) is np.ndarray:
                dF = self.c
            else:
                dF = self.c*np.ones_like(x)
            result = (result, dF)
        return result

    def prox(self, tau):
        self._prox.b = -tau
        return self._prox

class ConstFct(AffineFct):
    """ F(x) = const. """
    def __init__(self, N, const=0, conj=None):
        AffineFct.__init__(self, N, b=const, c=0)

class ZeroFct(ConstFct):
    """ F(x) = 0 """
    def __init__(self, N, conj=None):
        ConstFct.__init__(self, N, const=0)

    def prox(self, tau):
        # independent of tau
        return IdentityOp(self.x.size)

class MaskedAffineFct(Functional):
    """ F(x) = sum(c[mask,:]*x[mask,:]) + \delta_{x[not(mask),:] == 0} """
    def __init__(self, mask, c, conj=None):
        Functional.__init__(self)
        self.x = Variable(c.shape)
        self.mask = mask
        self.nmask = np.logical_not(mask)
        self.c = c
        if conj is None:
            from opymize.functionals import MaskedIndicatorFct
            self.conj = MaskedIndicatorFct(mask, c, conj=self)
        else:
            self.conj = conj
        scale = self.x.vars(self.x.new())[0]
        scale[self.mask,:] = 1.0
        self._prox = ShiftScaleOp(self.x.size, self.c.ravel(), scale.ravel(), -1)

    def __call__(self, x, grad=False):
        x = self.x.vars(x)[0]
        val = np.einsum('ik,ik->', x[self.mask,:], self.c[self.mask,:])
        infeas = norm(x[self.nmask,:], ord=np.inf)
        result = (val, infeas)
        if grad:
            dF = self.x.new()
            dF[self.mask,:] = self.c[self.mask,:]
            result = (result, dF.ravel)
        return result

    def prox(self, tau):
        self._prox.b = -tau
        return self._prox
