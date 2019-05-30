
import numpy as np

from opymize.operators.proj import QuadEpiProj
from opymize.tools.tests import test_gpu_op

tol = 1e-10

for i in range(10):
    N = np.random.randint(1,10)
    M = np.random.randint(1,5)
    a = 20*np.random.rand()
    b = -5.0 + 10*np.random.rand(N, M)
    c = -5.0 + 10*np.random.rand(N,)
    f = lambda _x: (0.5*a*_x**2 + b*_x).sum(axis=1) + c
    op = QuadEpiProj(N, M, alph=a, b=b, c=c)
    test_gpu_op(op)

    x = op.x.new().reshape(N, -1)
    x[:,0:-1] = np.random.rand(N, M)
    fx = f(x[:,0:-1])
    x[:,-1] = fx - 3*np.random.rand(N)
    op(x.ravel())
    assert np.all(np.abs(f(x[:,0:-1]) - x[:,-1]) < tol)

    x[:,0:-1] = np.random.rand(N, M)
    fx = f(x[:,0:-1])
    x[:,-1] = fx + np.random.rand(N)
    x = x.ravel()
    y = op.y.new()
    op(x, y)
    assert np.linalg.norm(x - y) < tol
