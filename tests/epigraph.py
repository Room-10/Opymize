
import numpy as np
np.set_printoptions(precision=4, linewidth=200, suppress=True, threshold=10000)

from opymize.operators import EpigraphProj
from opymize.functionals import EpigraphSupportFct
from opymize.tools.tests import test_gpu_op

tol = 1e-10

def test_proj_nd(ndim):
    npoints = 3
    nfuns = 2
    nregions = 1
    nsubpoints = npoints

    I = np.zeros((nfuns, npoints), dtype=bool)
    J = np.zeros((nregions, nsubpoints), dtype=np.int64)
    I[:] = [[ True, False, False],
            [False,  True,  True]]
    J[:] = np.arange(npoints)[None,:]

    v = -2.0 + 4*np.random.rand(npoints, ndim)
    b = -2.0 + 4*np.random.rand(nfuns, npoints)

    x_in = np.zeros((nregions, nfuns, ndim+1))
    x_out = x_in.copy()
    x = x_in.copy()

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
    x[0,:,-1] += 4*np.random.rand(nfuns)
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

def test_support_1d(nsubpoints):
    ndim = 1
    nfuns = 5
    nregions = 2
    npoints = nregions*nsubpoints

    fct = lambda x, _a, _b, _c: _a*(x - _b)**2 + _c
    x0 = 5*(-1 + 2*np.random.rand(nregions))
    x1 = x0 + 0.1 + 2*np.random.rand(nregions)
    a_arr = 0.1 + 4*np.random.rand(nfuns)
    b_arr = x0[0] + (-1.5 + 4*np.random.rand(nfuns))*(x1[0] - x0[0])
    c_arr = 5*(-1 + 2*np.random.rand(nfuns))

    I = np.ones((nfuns, npoints), dtype=bool)
    J = np.zeros((nregions, nsubpoints), dtype=np.int64)
    J[:] = np.arange(npoints).reshape(J.shape)

    v = np.zeros((nregions, nsubpoints, ndim), dtype=np.float64)
    for j in range(nregions):
        v[j,:] = np.linspace(x0[j], x1[j], nsubpoints)[:,None]
        tmp = J[j,-1]
        J[j,2:] = J[j,1:-1]
        J[j,1] = tmp
    v = v.reshape(npoints, ndim)

    b = np.zeros((nfuns, npoints), dtype=np.float64)
    for i in range(nfuns):
        b[i,:] = fct(v, a_arr[i], b_arr[i], c_arr[i]).ravel()

    faces = np.array([(0,2),(nsubpoints-1,1)] \
                   + [(k,k+1) for k in range(2,nsubpoints-1)])
    If = [[faces.copy() for j in range(nregions)] for i in range(nfuns)]

    sigma = EpigraphSupportFct(I, If, J, v, b)
    x = np.zeros((nregions, nfuns, ndim+1))
    for j in range(nregions):
        x[j,:,-1] = -np.random.rand()
        x[j,:,0] = -x[j,:,-1]*(x0[j] + np.random.rand(nfuns)*(x1[j] - x0[j]))
    f = np.zeros((nfuns, nregions))
    for i in range(nfuns):
        for j in range(nregions):
            xx, xt = x[j,i,:-1], x[j,i,-1]
            f[i,j] = -xt*fct(-xx/xt, a_arr[i], b_arr[i], c_arr[i])[0]
    sx = sigma(x)
    print("%.2e (%.2e)" % (np.abs(f.sum() - sx[0]), sx[1]))

def main():
    print("Testing EpigraphSupport (error should decrease, infeas=0) ...")
    for N in [5, 50, 500, 5000]:
        test_support_1d(N)
    print("Testing epigraph projections...")
    for i in range(10):
        test_proj_nd(2)
        test_proj_nd(3)

if __name__ == "__main__":
    main()
