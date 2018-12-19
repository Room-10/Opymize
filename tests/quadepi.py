
import numpy as np

from opymize.operators.proj import QuadEpiProj
from opymize.tools.tests import test_gpu_op

tol = 1e-10

for i in range(10):
    lbd = 20*np.random.rand()
    N = np.random.randint(1,10)
    M = np.random.randint(1,5)
    f = lambda x1: 0.5/lbd*np.sum(x1**2, axis=1)
    op = QuadEpiProj(N, M, lbd)
    test_gpu_op(op)

    x = op.x.new().reshape(N, -1)
    x[:,0:-1] = np.random.rand(N, M)
    fx = f(x[:,0:-1])
    x[:,-1] = fx*(-2 + 3*np.random.rand(N))
    op(x.ravel())
    assert np.all(np.abs(f(x[:,0:-1]) - x[:,-1]) < tol)

    x[:,0:-1] = np.random.rand(N, M)
    fx = f(x[:,0:-1])
    x[:,-1] = fx*(1 + np.random.rand(N))
    x = x.ravel()
    y = op.y.new()
    op(x, y)
    assert np.linalg.norm(x - y) < tol