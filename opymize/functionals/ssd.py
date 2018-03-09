
from opymize import Variable, Functional
from opymize.operators.affine import ShiftScaleOp

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
