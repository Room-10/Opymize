
from opymize import Variable, Operator
from opymize.linear.diff import GradientOp, DivergenceOp
from opymize.linear.einsum import *
from opymize.linear.scale import *

import numpy as np
from numpy.linalg import norm

def normest(linop, xy=None, tol=1.0e-4, maxits=500):
    """
    Estimate the spectral norm of the given linear operator `linop`.
    """
    m, n = linop.y.size, linop.x.size
    itn = 0

    # Compute an estimate of the abs-val column sums..
    x, y = (np.empty(n), np.empty(m)) if xy is None else xy
    y[:] = 1.0
    y[np.random.randn(m) < 0] = -1
    linop.adjoint(y, x)
    x[:] = np.abs(x)

    # Normalize the starting vector.
    e = norm(x)
    if e < 1e-13:
        return e, itn
    x *= 1.0/e
    e0 = 0
    while abs(e-e0) > tol*e:
        e0 = e
        linop(x, y)
        normy = norm(y)
        if normy < 1e-13:
            y[:] = np.random.rand(m)
            normy = norm(y)
        linop.adjoint(y, x)
        normx = norm(x)
        e = normx/normy
        x *= 1.0/normx
        itn += 1
        if itn > maxits:
            print("Warning: normest didn't converge!")
            break
    return e, itn

class BlockOp(LinOp):
    """ If A = [[A11,A12,...],[A21,A22,...],...], then
        A*(x1,x2,...) = (y1,y2,...)
        where yi = \sum_j Aij*xj
    """
    def __init__(self, blocks, adjoint=None):
        LinOp.__init__(self)
        self._call_cpu = self._call_gpu = self._call

        # determine size of blocks
        x_sizes = [None]*len(blocks[0])
        y_sizes = [None]*len(blocks)
        for i,Bi in enumerate(blocks):
            for j,Bij in enumerate(Bi):
                try:
                    x_sizes[j] = (Bij.x.size,)
                    y_sizes[i] = (Bij.y.size,)
                except AttributeError:
                    pass
        self.x = Variable(*x_sizes)
        self.y = Variable(*y_sizes)

        # replace 0 by ZeroOp
        for i,Bi in enumerate(blocks):
            for j,Bij in enumerate(Bi):
                if Bij == 0:
                    blocks[i][j] = ZeroOp(y_sizes[i], x_sizes[j])
        self.blocks = blocks

        # build adjoint matrix
        adj_bl = []
        for j,_ in enumerate(blocks[0]):
            adj_bl.append([b[j].adjoint for b in self.blocks])
        self.adjoint = BlockOp(adj_bl, adjoint=self) if adjoint is None else adjoint

    def prepare_gpu(self):
        for i,Bi in enumerate(self.blocks):
            for j,Bij in enumerate(Bi):
                Bij.prepare_gpu()

    def _call(self, x, y=None, add=False):
        yy = x.copy() if y is None else y
        if not add: yy.fill(0.0)
        for j,yj in enumerate(self.y.vars(yy)):
            for i,xi in enumerate(self.x.vars(x)):
                self.blocks[j][i](xi, yj, add=True)
        if y is None: x[:] = yy

    def rowwise_lp(self, y, p=1, add=False):
        if not add: y.fill(0.0)
        for j,yj in enumerate(self.y.vars(y)):
            for i,xi in enumerate(self.x.vars()):
                self.blocks[j][i].rowwise_lp(yj, p=p, add=True)

class CatOps(LinOp):
    """ ops = [A1,A2,A3] then Ax = A1*A2*A3*x """
    def __init__(self, ops, adjoint=None):
        LinOp.__init__(self)
        self.x = ops[0].x
        self.y = ops[-1].y
        self._tmp = [op.y.new() for op in ops[:-1]]
        self.ops = ops
        if adjoint is None:
            self.adjoint = CatOps([op.adjoint for op in reversed(ops)], adjoint=self)
        else:
            self.adjoint = adjoint

    def _call_cpu(self, x, y=None, add=False):
        yk = [x] + self._tmp + [y]
        for k,op in enumerate(self.ops):
            op(yk[k], yk[k+1], add=False if k+1 < len(self.ops) else add)
