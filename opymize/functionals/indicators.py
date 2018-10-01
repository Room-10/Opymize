
from opymize import Variable, Functional
from opymize.operators import ConstOp, ConstrainOp, PosProj, NegProj, EpigraphProj

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

    Cannot be called!
    """
    def __init__(self, I, J, v, b, conj=None):
        """
        Args:
            I : ndarray of bools, shape (nfuns, npoints)
            J : ndarray of ints, shape (nregions, nsubpoints)
            v : ndarray of floats, shape (npoints, 2)
            b : ndarray of floats, shape (nfuns, npoints)
        """
        Functional.__init__(self)

        nfuns, npoints = I.shape
        nregions, nsubpoints = J.shape

        self.x = Variable((nregions, nfuns, 3))

        if conj is None:
            self.conj = EpigraphSupportFct(I, J, v, b, conj=self)
        else:
            self.conj = conj

        self._prox = EpigraphProj(I, J, v, b)

    def prox(self, tau):
        # independent of tau
        return self._prox

class EpigraphSupportFct(Functional):
    """ F(x) = sum_ji max <y,x[j,i]> s.t. y \in epi(f[i]_j*)

    Cannot be called! prox is not implemented!
    """
    def __init__(self, I, J, v, b, conj=None):
        """
        Args:
            I : ndarray of bools, shape (nfuns, npoints)
            J : ndarray of ints, shape (nregions, nsubpoints)
            v : ndarray of floats, shape (npoints, 2)
            b : ndarray of floats, shape (nfuns, npoints)
        """
        Functional.__init__(self)

        nfuns, npoints = I.shape
        nregions, nsubpoints = J.shape

        self.x = Variable((nregions, nfuns, 3))

        if conj is None:
            self.conj = EpigraphFct(I, J, v, b, conj=self)
        else:
            self.conj = conj
