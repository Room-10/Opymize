Opymize
=======

A python package for formulating and solving non-smooth convex optimization
problems with objective functions of the form G(x) + F(Ax) where the convex
conjugates as well as proximal operators of the functionals G and F are
explicitly known while the linear operator A (along with its adjoint operator)
suffices to be known in matrix-free form.

In cases where the variable x consists of several block parts x=(x1,x2,...)
Opymize's interface is able to describe block linear operators A. For example:

    from opymize import LinOp
    from opymize.linear import BlockOp

    # set up A11, A12, A21, A22 ...
    assert all(type(op) is LinOp for op in [A11, A12, A21, A22])

    # compute the block matrix product using BlockOp
    A = BlockOp([[A11, A12],
                 [A21, A22]])
    x, Ax = A.x.new(), A.y.new()
    x[:] = np.random.randn(*x.shape)
    A(x, Ax)

    # compute the block matrix product manually
    y = A.y.new()
    x1, x2 = A.x.vars(x)
    y1, y2 = A.y.vars(y)
    A11(x1, y1)
    A12(x2, y1, add=True)
    A21(x1, y2)
    A22(x2, y2, add=True)

    # result should coincide
    assert np.allclose(Ax, y)

Similarly, it's possible to describe block functionals G(x) = \sum_i Gi(xi).

Currently, the only supported solver is a PDHG solver with constant, adaptive or
preconditioned step sizes. If all parts of the objective function are
implemented as CUDA kernels, it's possible to run the optimization on the GPU.

    from opymize import Functional, LinOp
    from opymize.solvers import PDHG

    # set up G, F and A ...
    assert type(G) is Functional
    assert type(F) is Functional
    assert type(A) is LinOp

    solver = PDHG(G, F, A)
    solver.solve(use_gpu=True, steps='adaptive')
    # result is stored in `solver.state` as primal-dual pair (x,y)

Setup
-----

Installing is as easy as

    pip install git+https://github.com/room-10/Opymize

For GPU support you need to install NVIDIA drivers and PyCuda manually.

To get started, have a look at the examples in the `examples` directory.
