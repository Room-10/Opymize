
import numpy as np

from opymize.linear.diff import GradientOp, LaplacianOp
from opymize.tools.tests import test_adjoint, test_rowwise_lp, test_gpu_op

domain = np.array([[-np.pi,np.pi],[-np.pi,np.pi]])

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
    grid = np.mgrid[[slice(0.0,s) for s in shape]].reshape(ndims, -1).T
    grid *= h[None,:]
    grid += domain[None,:,0] + 0.5*h[None,:]
    return grid, h

print("=> Testing gradient operator...")
imagedims = (10,12)
ndims = len(imagedims)
nchannels = 3
grad = GradientOp(imagedims, nchannels)
for op in [grad,grad.adjoint]:
    test_adjoint(op)
    test_rowwise_lp(op)
    test_gpu_op(op)

nchannels = 1
for s in range(1,4):
    imagedims = (10**s,10**s)
    grid, imageh = cell_centered_grid(domain, imagedims)
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

print("=> Testing Laplacian operator...")
for bdry in ["neumann","curvature"]:
    imagedims = (10,12)
    ndims = len(imagedims)
    nchannels = 3
    lplcn = LaplacianOp(imagedims, nchannels, boundary=bdry)
    for op in [lplcn,lplcn.adjoint]:
        test_adjoint(op)
        test_rowwise_lp(op)
        test_gpu_op(op)

    nchannels = 1
    for s in range(1,4):
        imagedims = (10**s,10**s)
        grid, imageh = cell_centered_grid(domain, imagedims)
        vol = np.prod(imageh)
        lplcn = LaplacianOp(imagedims, nchannels, imageh=imageh, boundary=bdry)
        y = lplcn.y.new()
        x = testfun(grid).ravel()
        lplcn(x, y)
        x, y = [v.reshape(imagedims) for v in [x,y]]
        ytest = testfun_laplacian(grid).reshape(imagedims)
        dif = y[1:-1,1:-1] - ytest[1:-1,1:-1]
        assert np.linalg.norm(vol*dif.ravel()) < 2*vol
