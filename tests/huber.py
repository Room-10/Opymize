
import numpy as np

from opymize.functionals import HuberPerspective

tol = 1e-10

for i in range(10):
    N = np.random.randint(1,10)
    M = np.random.randint(1,5)
    lbd = 0.01 + 6*np.random.rand()
    alph = 0.01 + 3*np.random.rand()

    F = HuberPerspective(N, M, lbd=lbd, alph=alph)
    x = F.x.new().reshape(N, M+1)
    x[:,:-1] = 5*np.random.randn(N, M)
    x[:,-1] = -1

    # Check positive one-homogeneity
    fx, infeas = F(x.ravel())
    assert np.abs(infeas) < tol
    fact = 0.001 + 5*np.random.rand()
    x *= fact
    fx2, infeas = F(x.ravel())
    assert np.abs(infeas) < tol
    assert np.abs(fact*fx - fx2) < tol

    # Check infeasible set
    x[:,-1] = 0.5*np.random.rand()
    fx, infeas = F(x.ravel())
    assert np.abs(fx) < tol

    # Check (radial) linearity outside of alph-ball
    rsamples = alph + 2*np.random.rand(N,1)
    x[:,:-1] *= rsamples/np.linalg.norm(x[:,:-1], axis=-1)[:,None]
    x[:,-1] = -1
    fx, infeas = F(x.ravel())
    assert np.abs(fx - lbd*(rsamples - 0.5*alph).sum()) < tol