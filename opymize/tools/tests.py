
from opymize.linear import normest

import numpy as np
from numpy.linalg import norm

def checkFctDerivative(fun, x):
    """
    Check gradient of `fun` at point `x`.
    """
    (fx, _), gradfx = fun(x, grad=True)
    m0 = 1
    N = 8
    for m in range(m0,m0+N):
        h = 10**(-m)
        err = 0
        for i in range(20):
            ve = np.random.randn(x.size)
            ve /= np.sqrt(np.sum(ve**2))
            v = h*ve
            taylor = fx + np.einsum('i,i->', gradfx, v)
            err = max(err, np.abs(fun(x + v)[0] - taylor))
        print('%02d: % 7.2e % 7.2e % 7.2e' % (m, h, err, err/h**2))

def checkOpDerivative(op, x):
    """
    Check jacobian of `op` at point `x`.
    """
    Tx = op.y.new()
    Txv = Tx.copy()
    dTx = op(x, Tx, jacobian=True)
    dTx_v = dTx.y.new()
    m0 = 1
    N = 8
    for m in range(m0,m0+N):
        h = 10**(-m)
        err = 0
        for i in range(20):
            ve = np.random.randn(x.size)
            ve /= np.sqrt(np.sum(ve**2))
            v = h*ve
            dTx(v, dTx_v)
            op(x + v, Txv, jacobian=False)
            err = max(err, norm(Txv - Tx - dTx_v, ord=2))
        print('%02d: % 7.2e % 7.2e % 7.2e' % (m, h, err, err/h**2))

def test_adjoint(op):
    x = np.random.randn(op.x.size)
    y = np.random.randn(op.y.size)
    ATy = 0*x
    Ax = 0*y
    op(x, Ax)
    op.adjoint(y, ATy)
    assert(np.abs(np.sum(Ax*y) - np.sum(x*ATy)) < 1e-10)
    print("adjoint operator tested successfully")

def test_rowwise_lp(op, p=1, maxiter=10):
    for _ in range(maxiter):
        i = np.random.randint(0, op.y.size)
        y = op.y.new()
        y[i] = 1.0
        Ai = op.x.new()
        op.adjoint(y, Ai)
        op.rowwise_lp(y, p=p)
        assert(np.abs(y[i] - np.sum(np.abs(Ai)**p)) < 1e-10)
    print("rowwise l^p tested successfully")

def test_gpu_op(op, type_t="double"):
    import opymize.tools.gpu
    from pycuda import gpuarray

    np_dtype = np.float64 if type_t == "double" else np.float32
    x, Ax = op.x.new(dtype=np_dtype), op.y.new(dtype=np_dtype)

    x[:], Ax[:] = [np.random.randn(v.size) for v in [op.x, op.y]]
    x_gpu, Ax_gpu = [gpuarray.to_gpu(v) for v in [x, Ax]]

    op.prepare_gpu(type_t=type_t)
    op(x, Ax)
    op(x_gpu, Ax_gpu)

    print("Bootstrap test for GPU implementation of %s" % type(op))
    compare_vars(op.y, Ax, Ax_gpu.get())

def compare_vars(descr, v1, v2):
    for i,(v1i,v2i) in enumerate(zip(descr.vars(v1), descr.vars(v2))):
        print("Testing i=%d ... " % i, end="")
        if not np.allclose(v1i, v2i):
            raise Exception("Mismatch: %e" % np.amax(np.abs(v1i-v2i)))
        print("successful!")
