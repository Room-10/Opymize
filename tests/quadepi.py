
import numpy as np

from opymize.operators.proj import QuadEpiProj
from opymize.tools.tests import test_gpu_op

tol = 1e-10

def sample_unit_ball(N, M):
    sample = sample_unit_sphere(N, M)
    sample *= np.random.rand(N,1)**(1/M)
    return sample

def sample_unit_sphere(N, M):
    norms = np.zeros(1)
    while np.any(norms == 0):
        vsample = np.random.randn(N, M)
        norms = np.linalg.norm(vsample, axis=-1)
    return vsample/norms[:,None]

def test_quadepiproj_inside(f, op, lbd=10):
    N = op.x[0]['shape'][0]
    M = op.x[0]['shape'][1] - 1
    x = op.x.new().reshape(N, -1)
    x[:,:-1] = lbd*sample_unit_ball(N, M)
    fx = f(x[:,:-1])
    x[:,-1] = fx + np.random.rand(N)
    x = x.ravel()
    y = op.y.new()
    op(x, y)
    assert np.linalg.norm(x - y) < tol

def test_quadepiproj_outside(f, op, lbd=10):
    N = op.x[0]['shape'][0]
    M = op.x[0]['shape'][1] - 1
    x = op.x.new().reshape(N, -1)
    x[:,:-1] = lbd*sample_unit_ball(N, M)
    fx = f(x[:,:-1])
    x[:,-1] = fx - 3*np.random.rand(N)
    op(x.ravel())
    assert np.all(np.abs(f(x[:,:-1]) - x[:,-1]) < tol)

def test_quadepiproj_trunc_outside(f, op, lbd):
    N = op.x[0]['shape'][0]
    M = op.x[0]['shape'][1] - 1
    x = op.x.new().reshape(N, -1)
    x[:,:-1] = lbd*sample_unit_sphere(N, M)
    fx = f(x[:,:-1])
    x[:,:-1] *= (1 + np.random.rand(N,1))
    x[:,-1] = fx + 3*np.random.rand(N)
    op(x.ravel())
    x1norms = np.linalg.norm(x[:,:-1], axis=-1)
    assert np.all(fx - x[:,-1] < tol)
    assert np.all(np.abs(x1norms - lbd).max() < tol)

def test_quadepiproj_trunc_outside2(f, op, lbd):
    N = op.x[0]['shape'][0]
    M = op.x[0]['shape'][1] - 1
    x = op.x.new().reshape(N, -1)
    x[:,:-1] = lbd*sample_unit_sphere(N, M)
    fx = f(x[:,:-1])
    alph = 2*fx[0]/lbd**2
    x1facts = (1 + np.random.rand(N))
    x[:,:-1] *= x1facts[:,None]
    x[:,-1] = fx - np.random.rand(N)*(x1facts - 1)/alph
    op(x.ravel())
    x1norms = np.linalg.norm(x[:,:-1], axis=-1)
    assert np.all(np.abs(fx - x[:,-1]).max() < tol)
    assert np.all(np.abs(x1norms - lbd).max() < tol)

def test_quadepiproj_trunc_outside3(f, op, lbd):
    N = op.x[0]['shape'][0]
    M = op.x[0]['shape'][1] - 1
    x = op.x.new().reshape(N, -1)
    x[:,:-1] = lbd*sample_unit_sphere(N, M)
    fx = f(x[:,:-1])
    alph = 2*fx[0]/lbd**2
    x1facts = (1 + np.random.rand(N))
    x[:,:-1] *= x1facts[:,None]
    x[:,-1] = fx - (1 + 3*np.random.rand(N))*(x1facts - 1)/alph
    op(x.ravel())
    assert np.all(np.abs(f(x[:,:-1]) - x[:,-1]) < tol)

def test_quadepiproj_abc():
    N = np.random.randint(1,10)
    M = np.random.randint(1,5)
    a = 20*np.random.rand()
    b = -5.0 + 10*np.random.rand(N, M)
    c = -5.0 + 10*np.random.rand(N,)
    f = lambda _x: (0.5*a*_x**2 + b*_x).sum(axis=1) + c
    op = QuadEpiProj(N, M, alph=a, b=b, c=c)
    test_gpu_op(op)
    test_quadepiproj_outside(f, op)
    test_quadepiproj_inside(f, op)

def test_quadepiproj_shift():
    N = np.random.randint(1,10)
    M = np.random.randint(1,5)
    a = 20*np.random.rand()
    shift = -5.0 + 10*np.random.rand(N, M+1)
    f = lambda _x: 0.5*a*((_x - shift[:,:-1])**2).sum(axis=1) + shift[:,-1]
    op = QuadEpiProj(N, M, alph=a, shift=shift)
    test_gpu_op(op)
    test_quadepiproj_inside(f, op)
    test_quadepiproj_outside(f, op)

def test_quadepiproj_trunc():
    N = np.random.randint(1,10)
    M = np.random.randint(1,5)
    a = 20*np.random.rand()
    lbd = 5*np.random.rand()
    f = lambda _x: 0.5*a*(_x**2).sum(axis=1)
    op = QuadEpiProj(N, M, alph=a, lbd=lbd)
    test_gpu_op(op)
    test_quadepiproj_inside(f, op, lbd=lbd)
    test_quadepiproj_outside(f, op, lbd=lbd)
    test_quadepiproj_trunc_outside(f, op, lbd)
    test_quadepiproj_trunc_outside2(f, op, lbd)
    test_quadepiproj_trunc_outside3(f, op, lbd)

def main():
    for i in range(10):
        test_quadepiproj_abc()
        test_quadepiproj_shift()
        test_quadepiproj_trunc()

if __name__ == "__main__":
    main()
