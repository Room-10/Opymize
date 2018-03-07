
from opymize import Variable, Functional
from opymize.operators import SplitOp
from opymize.functionals.affine import *
from opymize.functionals.indicators import *
from opymize.functionals.l1norms import *
from opymize.functionals.ssd import *

class SplitSum(Functional):
    """ F(x1,x2,...) = F1(x1) + F2(x2) + ... """
    def __init__(self, fcts, conj=None):
        Functional.__init__(self)
        self.x = Variable(*[(F.x.size,) for F in fcts])
        self.fcts = fcts
        self.conj = SplitSum([F.conj for F in fcts], conj=self) if conj is None else conj

    def __call__(self, x, grad=False):
        X = self.x.vars(x)
        results = [F(xi, grad=grad) for F,xi in zip(self.fcts, X)]
        if grad:
            val = sum([res[0][0] for res in results])
            infeas = sum([res[0][1] for res in results])
            dF = np.concatenate([res[1] for res in results])
            return (val, infeas), dF
        else:
            val = sum([res[0] for res in results])
            infeas = sum([res[1] for res in results])
            return (val, infeas)

    def prox(self, tau):
        if type(tau) is np.ndarray:
            tau = self.x.vars(tau)
            prox_ops = []
            for F,Ftau in zip(self.fcts,tau):
                prox_ops.append(F.prox(Ftau))
            return SplitOp(prox_ops)
        else:
            return SplitOp([F.prox(tau) for F in self.fcts])