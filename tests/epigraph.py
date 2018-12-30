
import numpy as np
import cvxopt

from opymize.operators.proj import EpigraphProj
from opymize.tools.tests import test_gpu_op

tol = 1e-10
cvxopt.solvers.options['reltol'] = 1e-2*tol
cvxopt.solvers.options['abstol'] = 1e-2*tol

ndim = 2
npoints = 3
nfuns = 2
nregions = 1
nsubpoints = npoints
I = np.array([[ True, False, False],
              [False,  True,  True]], dtype=bool)
J = np.arange(npoints)[None,:]
b = np.zeros((nfuns,npoints))
v = np.zeros((npoints, ndim))
x_in = np.zeros((nregions, nfuns, ndim+1))
x_out = x_in.copy()
x = x_in.copy()

for i in range(10):
    v = -2.0 + 4*np.random.rand(*v.shape)
    b = -2.0 + 4*np.random.rand(*b.shape)
    op = EpigraphProj(I, J, v, b)
    op.cp_tol = tol*1e-3

    # x_in[0,0] : point on the plane (boundary of half space)
    x_in[0,0,:] = -5.0 + 10*np.random.rand(ndim+1)
    x_in[0,0,-1] = -b[0,0] + v[0].dot(x_in[0,0,:-1])

    # x_in[0,1] : point on the boundary
    x_in[0,1,:] = -5.0 + 10*np.random.rand(ndim+1)
    vals = [-bk + vk.dot(x_in[0,1,:-1]) for bk,vk in zip(b[1,1:],v[1:])]
    active_k = np.argmax(vals)
    x_in[0,1,-1] = vals[active_k]

    # x : point inside half space (mapped to itself)
    x = x_in.copy()
    x[0,:,-1] += 4*np.random.rand(ndim)
    op(x, x_out)
    assert np.linalg.norm(x_out - x)**2 < tol

    # x : point outside half space (mapped back to boundary)
    x = x_in.copy()
    shift = np.random.rand(2)
    x[0,0,:-1] += shift[0]*v[0]
    x[0,0,-1] -= shift[0]
    x[0,1,:-1] += shift[1]*v[1+active_k]
    x[0,1,-1] -= shift[1]
    op(x, x_out)
    assert np.linalg.norm(x_out - x_in)**2 < tol

    test_gpu_op(op)