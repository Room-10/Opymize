
from opymize import Operator
from opymize.operators.affine import *
from opymize.operators.proj import *

class SplitOp(Operator):
    """ T(x1,x2,...) = (T1(x1),T2(x2),...) """
    def __init__(self, ops):
        Operator.__init__(self)
        self.x = Variable(*[(T.x.size,) for T in ops])
        self.y = Variable(*[(T.y.size,) for T in ops])
        self.ops = ops
        self._call_cpu = self._call_gpu = self._call

    def prepare_gpu(self):
        [op.prepare_gpu() for op in self.ops]

    def _call(self, x, y=None, add=False, jacobian=False):
        X = self.x.vars(x)
        if y is None:
            Y = [None for xi in X]
        else:
            Y = self.y.vars(y)
        results = [[0]*len(self.ops) for T in self.ops]
        for i, (T, xi, yi) in enumerate(zip(self.ops, X, Y)):
            results[i][i] = T(xi, yi, add=add, jacobian=jacobian)
        if jacobian: return BlockOp(results)
