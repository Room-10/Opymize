
from opymize import Variable, Functional
from opymize.operators.proj import L1NormsProj

import numpy as np
from numpy.linalg import norm

def norms(x, res, matrixnorm="nuclear"):
    """ Compute the norm of each x^i and store it in res.

    The nuclear norm is the dual norm of the spectral norm

    Args:
        x : numpy array of shape (N, M1, M2) where `M1 in (1,2) or M2 in (1,2)`
        res : numpy array of shape (N,)
        matrixnorm : one of 'nuclear', 'spectral' or 'frobenius'
    Returns:
        nothing, the result is written to res
    """
    N, M1, M2 = x.shape
    if matrixnorm == 'nuclear':
        matrixnorm = 'nuc'
    elif matrixnorm == 'spectral':
        matrixnorm = 2

    if M1 == 1 or M2 == 1 or matrixnorm == "frobenius":
        np.einsum('ilm,ilm->i', x, x, out=res)
        np.sqrt(res, out=res)
    elif M1 == 2 or M2 == 2:
        res[:] = norm(x, ord=matrixnorm, axis=(1,2))
    else:
        raise Exception("Dimension error: M1={:d}, M2={:d}".format(M1, M2))

class L1Norms(Functional):
    """ F(x) = lbd * \sum_i |x[i,:,:]|
    Supported norms are 'frobenius' and 'nuclear'
    """
    def __init__(self, N, M, lbd, matrixnorm="frobenius", conj=None):
        Functional.__init__(self)
        assert matrixnorm in ['frobenius', 'nuclear']
        self.x = Variable((N,) + M)
        self.lbd = lbd
        self.matrixnorm = matrixnorm
        conjnorm = 'spectral' if matrixnorm == 'nuclear' else 'frobenius'
        self.conj = L1NormsConj(N, M, lbd, conjnorm, conj=self) if conj is None else conj
        self._xnorms = np.zeros((N,), order='C')

    def __call__(self, x, grad=False):
        x = self.x.vars(x)[0]
        norms(x, self._xnorms, self.matrixnorm)
        val = self.lbd*self._xnorms.sum()
        infeas = 0
        result = (val, infeas)
        if grad:
            assert self.matrixnorm == 'frobenius'
            """ dF = 0 if x == 0 else x/|x| """
            dF = x.copy()
            where0 = (self._xnorms == 0)
            wheren0 = np.logical_not(where0)
            dF[where0,:,:] = 0
            dF[wheren0,:,:] /= self._xnorms[wheren0]
            result = result, dF.ravel()
        return result

class L1NormsConj(Functional):
    """ F(x) = \sum_i \delta_{|x[i,:,:]| \leq lbd}
    Supported norms are 'frobenius' and 'spectral'
    """
    def __init__(self, N, M, lbd, matrixnorm="frobenius", conj=None):
        Functional.__init__(self)
        assert matrixnorm in ['frobenius', 'spectral']
        self.x = Variable((N,) + M)
        self.lbd = lbd
        self.matrixnorm = matrixnorm
        conjnorm = 'nuclear' if matrixnorm == 'spectral' else 'frobenius'
        self.conj = L1Norms(N, M, lbd, conjnorm, conj=self) if conj is None else conj
        self._prox = L1NormsProj(N, M, self.lbd, matrixnorm)
        self._xnorms = np.zeros((N,), order='C')

    def __call__(self, x, grad=False):
        x = self.x.vars(x)[0]
        norms(x, self._xnorms, self.matrixnorm)
        val = 0
        infeas = norm(np.fmax(0, self._xnorms - self.lbd), ord=np.inf)
        result = (val, infeas)
        if grad:
            result = result, self.x.new()
        return result

    def prox(self, tau):
        # independent of tau!
        return self._prox
