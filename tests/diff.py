
import numpy as np
import scipy.sparse as sp
import cvxpy as cp

from opymize.linear.diff import GradientOp, LaplacianOp
from opymize.tools.tests import test_adjoint, test_rowwise_lp, test_gpu_op

testfun_domain = np.array([[-np.pi,np.pi],[-np.pi,np.pi]])

def testfun(pts):
    x, y = pts[:,0], pts[:,1]
    return np.sin(x) + x**2*np.cos(y)

def testfun_grad(pts):
    x, y = pts[:,0], pts[:,1]
    return np.vstack([np.cos(x) + 2*x*np.cos(y), -x**2*np.sin(y)]).T

def testfun_laplacian(pts):
    x, y = pts[:,0], pts[:,1]
    return (2 - x**2)*np.cos(y) - np.sin(x)

def cell_centered_grid(domain, shape):
    h = (domain[:,1] - domain[:,0])/shape
    ndims = len(shape)
    grid = np.mgrid[[slice(0.0,s) for s in shape]].reshape(ndims, -1).T
    grid *= h[None,:]
    grid += domain[None,:,0] + 0.5*h[None,:]
    return grid, h

def test_grad():
    imagedims = (10,12)
    imageh = np.array([0.3,0.4])
    ndims = len(imagedims)
    nchannels = 3
    grad = GradientOp(imagedims, nchannels, imageh=imageh)
    for op in [grad,grad.adjoint]:
        test_adjoint(op)
        test_rowwise_lp(op)
        test_gpu_op(op)

def test_grad_fun(s):
    imagedims = (10**s,10**s)
    ndims = len(imagedims)
    nchannels = 1
    grid, imageh = cell_centered_grid(testfun_domain, imagedims)
    grid2 = grid + 0.5*imageh[None,:]
    vol = np.prod(imageh)
    grad = GradientOp(imagedims, nchannels, imageh=imageh)
    y = grad.y.new()
    x = testfun(grid).ravel()
    grad(x, y)
    x = x.reshape(imagedims)
    y = y.reshape(imagedims + (ndims,))
    ytest = testfun_grad(grid2).reshape(imagedims + (ndims,))
    dif = y[:-1,:-1] - ytest[:-1,:-1]
    assert np.linalg.norm(vol*dif.ravel()) < 2*vol

def test_lplcn(bdry):
    imagedims = (10,12)
    ndims = len(imagedims)
    nchannels = 3
    lplcn = LaplacianOp(imagedims, nchannels, boundary=bdry)
    for op in [lplcn,lplcn.adjoint]:
        test_adjoint(op)
        test_rowwise_lp(op)
        test_gpu_op(op)

def test_lplcn_fun(bdry, s):
    imagedims = (10**s,10**s)
    ndims = len(imagedims)
    nchannels = 1
    grid, imageh = cell_centered_grid(testfun_domain, imagedims)
    vol = np.prod(imageh)
    lplcn = LaplacianOp(imagedims, nchannels, imageh=imageh, boundary=bdry)
    y = lplcn.y.new()
    x = testfun(grid).ravel()
    lplcn(x, y)
    x, y = [v.reshape(imagedims) for v in [x,y]]
    ytest = testfun_laplacian(grid).reshape(imagedims)
    dif = y[1:-1,1:-1] - ytest[1:-1,1:-1]
    assert np.linalg.norm(vol*dif.ravel()) < 2*vol

def test_lplcn_ghost():
    def laplop(m, n):
        ddn = sp.spdiags(np.ones(n)*np.array([[1, -2, 1]]).T, [-1, 0, 1], n, n)
        ddm = sp.spdiags(np.ones(m)*np.array([[1, -2, 1]]).T, [-1, 0, 1], m, m)
        return sp.kron(ddm, sp.eye(n,n)) + sp.kron(sp.eye(m,m), ddn)

    imagedims = np.array((30, 40))
    data = np.random.rand(*imagedims)

    op = LaplacianOp(imagedims, 1, boundary="curvature")
    Dy_curv = op.y.new().reshape(imagedims)
    op(data, Dy_curv)

    gimagedims = imagedims+2
    A = cp.Constant(laplop(*gimagedims[::-1]))
    y = cp.Variable(gimagedims)
    Dy = cp.reshape(A*cp.vec(y), gimagedims)
    cp.Problem(
        cp.Minimize(cp.sum_squares(Dy[1:-1,1:-1])),
        [y[1:-1,1:-1] == data]
    ).solve()
    Dy_ghost = Dy.value

    assert np.linalg.norm(Dy_curv - Dy_ghost[1:-1,1:-1], ord=np.inf) < 1e-12

if __name__ == "__main__":
    print("=> Testing gradient operator...")
    test_grad()
    for s in range(1,4):
        test_grad_fun(s)

    for bdry in ["curvature", "neumann", "second-order"]:
        print("=> Testing Laplacian operator with %s bc..." % bdry)
        test_lplcn(bdry)
        for s in range(1,4):
            test_lplcn_fun(bdry, s)
    test_lplcn_ghost()
