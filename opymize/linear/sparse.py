
import numpy as np
import scipy.sparse as sp

# sparse matrix representations of selected linear operations
# all functions assume 'C' order ravelling if not specified otherwise

def lplcnop2(dims, components=1, steps=None, boundaries="neumann"):
    """ Two-dimensional finite difference Laplacian operator

    Args:
        dims : pair of ints
            shape of the image domain
        components : int
            for vector-valued images
        boundaries : (optional)
            one of 'neumann' (default), 'dirichlet' or 'curvature'
    """
    if steps is None:
        steps = np.ones(len(dims))

    def dd(n):
        diags = np.ones(n)*np.array([[1, -2, 1]]).T
        if boundaries == "curvature":
            diags[[0,0,1,1,2,2],[-2,-1,0,-1,0,1]] = 0
        elif boundaries == "neumann":
            diags[[1,1],[0,-1]] = -1
        elif boundaries == "dirichlet":
            pass
        else:
            raise Exception("Unsupported boundary condition: %s" % boundaries)
        return sp.spdiags(diags, [-1, 0, 1], n, n)

    def eye(n):
        if boundaries == "curvature":
            diags = np.ones((1,n))
            diags[0,[0,-1]] = 0
            return sp.spdiags(diags, [0], n, n)
        else:
            return sp.eye(n)

    n1, n2 = dims
    Delta = sp.kron( dd(n1),eye(n2))/steps[0]**2 \
          + sp.kron(eye(n1), dd(n2))/steps[1]**2
    # einsum: 'ij,jk->ik'
    return sp.kron(Delta, sp.eye(components))

def diffopn(dims, components=1, steps=None, weights=None,
                  schemes="forward", boundaries="neumann"):
    """ Multidimensional finite difference operator

    Args:
        dims : tuple
            shape of the image domain
        components : int
            for vector-valued images
        steps : ndarray of floats, shape (ndims,)
            grid step sizes
        weights : ndarray of floats, shape (components,)
        schemes, boundaries : see diffop
    """
    if steps is None:
        steps = np.ones(len(dims))
    if weights is None:
        weights = np.ones(components)

    if schemes == "centered":
        # the `boundaries` are ignored!
        D = diffopn(dims, steps=steps, schemes="forward", boundaries="neumann")
        return sp.kron(avgopn(dims).dot(D), sp.diags(weights))

    if type(boundaries) is str:
        boundaries = len(dims)*(boundaries,)
    if type(schemes) is str:
        schemes = len(dims)*(schemes,)

    partials = []
    for t,size in enumerate(dims):
        partial_t = diffop(size, scheme=schemes[t],
                                 boundaries=boundaries[t])/steps[t]
        # einsum 'ij,kjl->kil'
        partial_t = extendedop(partial_t, before=dims[:t], after=dims[(t+1):])
        partials.append(partial_t)
    # einsum: 'tij,j->it'
    return sp.kron(stackedop(partials, order='F'), sp.diags(weights))

def diffop(size, scheme="forward", boundaries="neumann"):
    """ Finite difference matrix for 1D data

    Args:
        size : int
            data size
        scheme : (optional)
            one of 'forward' (default), 'backward', 'central'
        boundaries : (optional)
            one of 'neumann' (default), 'dirichlet', 'periodic', 'second-order',
            'third-order' or a pair of these for different boundary conditions
            left and right

    Returns:
        sparse 2D numpy array of shape (size,size)
    """
    if type(boundaries) is str:
        boundaries = 2*(boundaries,)

    diags = []
    indices = []
    if scheme == "forward":
        diags = [-np.ones(size), np.ones(size-1)]
        indices = [0, 1]
        if boundaries[1] == "neumann":
            diags[0][-1] = 0
        elif boundaries[1] == "dirichlet":
            pass
        elif boundaries[1] == "periodic":
            diags.append(np.ones(1))
            indices.append(-size+1)
        elif boundaries[1] == "second-order":
            diags.append(np.zeros(size-1))
            indices.append(-1)
            diags[0][-1] = 1
            diags[-1][-1] = -1
        elif boundaries[1] == "third-order":
            diags.extend([np.zeros(size-2), np.zeros(size-1)])
            indices.extend([-2, -1])
            diags[0][-1] = 2
            diags[-1][-1] = -3
            diags[-2][-1] = 1
        else:
            raise Exception("Unsupported boundary condition for this scheme")
    elif scheme == "backward":
        diags = [np.ones(size), -np.ones(size-1)]
        indices = [0, -1]
        if boundaries[0] == "neumann":
            diags[0][0] = 0
        elif boundaries[0] == "dirichlet":
            pass
        elif boundaries[0] == "periodic":
            diags.append(-np.ones(1))
            indices.append(size-1)
        elif boundaries[0] == "second-order":
            diags.append(np.zeros(size-1))
            indices.append(1)
            diags[0][0] = -1
            diags[-1][0] = 1
        elif boundaries[0] == "third-order":
            diags.extend([np.zeros(size-1), np.zeros(size-2)])
            indices.extend([1, 2])
            diags[0][0] = -2
            diags[-2][0] = 3
            diags[-1][0] = -1
        else:
            raise Exception("Unsupported boundary condition for this scheme")
    elif scheme == "central":
        diags = [np.zeros(size), np.ones(size-1), -np.ones(size-1)]
        indices = [0, 1, -1]

        if boundaries[0] == "neumann":
            diags[1][0] = 0
        elif boundaries[0] == "dirichlet":
            pass
        elif boundaries[0] == "periodic":
            diags.append(-np.ones(1))
            indices.append(size-1)
        elif boundaries[0] == "second-order":
            diags.append(np.zeros(size-2))
            indices.append(2)
            diags[0][0] = -1
            diags[1][0] = 0
            diags[-1][0] = 1
        elif boundaries[0] == "third-order":
            diags.extend([np.zeros(size-2), np.zeros(size-3)])
            indices.extend([2, 3])
            diags[0][0] = -2
            diags[1][0] = 1
            diags[-2][0] = 2
            diags[-1][0] = -1
        else:
            raise Exception("Unsupported boundary condition for this scheme")

        if boundaries[1] == "neumann":
            diags[2][-1] = 0
        elif boundaries[1] == "dirichlet":
            pass
        elif boundaries[1] == "periodic":
            diags.append(np.ones(1))
            indices.append(-size+1)
        elif boundaries[1] == "second-order":
            diags.append(np.zeros(size-2))
            indices.append(-2)
            diags[-1][-1] = -1
            diags[2][-1] = 0
            diags[0][-1] = 1
        elif boundaries[1] == "third-order":
            diags.extend([np.zeros(size-2), np.zeros(size-3)])
            indices.extend([-2, -3])
            diags[-1][-1] = 1
            diags[-2][-1] = -2
            diags[2][-1] = -1
            diags[0][-1] = 2
        else:
            raise Exception("Unsupported boundary condition for this scheme")
    else:
        raise Exception("Unsupported scheme")
    return sp.diags(diags, indices, shape=(size,size))

def avgopn(dims):
    """ Generate an operator that averages a tangent vector field in the cell
        centers along the specified dimensions with reflecting boundary.

    Args:
        dims : tuple
            shape of the image domain
    """
    dims = np.array(dims)

    avgs = []
    for t in range(dims.size):
        avg_t = 0
        for ti in range(dims.size):
            if t == ti:
                continue
            diags = np.ones((2, dims[ti]))
            diags[0,-1] = 0
            A = sp.diags(diags, [0, 1], shape=(dims[ti],dims[ti]))
            # einsum: kl,ilj->ikj
            A = extendedop(A, before=dims[:ti], after=dims[ti+1:])
            avg_t = 0.5*(avg_t + A)
        avgs.append(avg_t)
    # einsum: tij,jt->it
    return diagop(avgs, order=('F','F'))

def transposeopn(shape, axes):
    """ Sparse matrix representation of y = x.transpose(axes) """
    assert sorted(axes) == list(range(len(shape)))
    shape = np.asarray(shape, dtype=np.int64)
    ndims = shape.size
    N = np.prod(shape)
    if sum(shape > 1) <= 1 or axes == sorted(axes):
        return sp.eye(N)

    strides = np.zeros_like(shape)
    strides[-1] = 1
    for t in range(ndims-2,-1,-1):
        strides[t] = strides[t+1]*shape[t+1]

    strides, shape = [s[list(axes)] for s in [strides, shape]]
    idx = np.asarray(list(np.ndindex(*shape))).dot(strides).astype(np.int64)
    data = (np.ones(N), (range(N), idx))
    return sp.coo_matrix(data, shape=(N,N), dtype=np.int8)

def transposeop(m, n):
    return transposeopn((m,n), (1,0))

def idxop(P, m):
    # y[i] = x[P[i]]  where  P[i] \in {0,...,m}
    n = P.size
    return sp.coo_matrix((np.ones(n), (range(n),P)), shape=(n,m))

def diagop(As, order='C'):
    # y[j,l] = sum_k As[j][l,k]*x[j,k]
    if order in ['C','F']:
        order = (order, order)
    J, (L, K) = len(As), As[0].shape
    T_op_in = transposeop(K, J) if order[0] == 'F' else sp.eye(K*J)
    T_op_out = transposeop(J, L) if order[1] == 'F' else sp.eye(J*L)
    return T_op_out.dot(sp.block_diag(As)).dot(T_op_in)

def stackedop(As, order='C'):
    # y[j,l] = sum_k As[j][l,k]*x[k]
    J, (L, K) = len(As), As[0].shape
    T_op_out = transposeop(J, L) if order == 'F' else sp.eye(J*L)
    return T_op_out.dot(sp.vstack(As))

def extendedop(A, before=None, after=None):
    """ y[...,i,...] = sum_j A[i,j]*x[...,j,...] """
    if before is not None and len(before) > 0:
        # block matrix with A repeated `before[-1]` times along the diagonal
        A = sp.kron(sp.eye(before[-1]), A)
        A = extendedop(A, before=before[:-1], after=after)
    elif after is not None and len(after) > 0:
        # each entry of A is stretched to a diagonal block of size `after[0]`
        A = sp.kron(A, sp.eye(after[0]))
        A = extendedop(A, after=after[1:])
    return A

def array_shape(data):
    try:
        return data.shape
    except:
        try:
            return (len(data),) + data_shape(data[0])
        except:
            return ()

def derive_shape(dout, din, sin):
    sout = [0,]*len(dout)
    for j,dn in enumerate(dout):
        iin = np.array([d.find(dn) for d in din], dtype=np.int64)
        i = np.where(iin >= 0)[0][0]
        sout[j] = sin[i][iin[i]]
    return tuple(sout)

def einsumop(subscripts, data, shape_in, shape_out=None, order='C'):
    """ Generate sparse operator from numpy.einsum subscripts

    For `subscripts = ",ij->ji"` and `data = 1` the result is transposeop.

    For indices on the rhs that don't appear on the lhs (as in
    `subscripts = ",->i"`), a tiled matrix is generated.
    In this case, `shape_out` is mandatory.

    Args:
        subscripts : string
            tensor dot string, see documention of numpy.einsum
        data : array-like object
            Can be a list of sparse matrices or a scalar (for 0-dimensional)
            or an ndarray or a list of ndarrays etc.
        shape_in : tuple of ints
        shape_out : tuple of ints
            If None, the output shape is determined from `subscripts`, `data`
            and `shape_in`.
        order : {'C','F'} or pair of these
            Assume that input (and output) have specified ordering.

    Examples:
        >>> subscripts = "ijkl,kij->lj"
        >>> A = einsumop(subscripts, data, shape_in, shape_out)
        >>> # suppose `x` is array of shape `shape_in`
        >>> assert np.all(A.dot(x) == np.einsum(subscripts, data, x).ravel())
    """
    # data is sparse matrix: use direct (instead of recursive) approach
    if sp.issparse(data):
        # see diagop, stackedop and extendedop for selected cases that
        # are already available
        raise Exception("Not implemented!")

    din, dout = subscripts.split("->")
    ddata, din = din.split(",")
    if order in ['C','F']:
        order = (order, order)
    din = din[::-1] if order[0] == 'F' else din
    dout = dout[::-1] if order[1] == 'F' else dout

    sdata = array_shape(data)
    sin, sout = shape_in, shape_out
    if sout is None:
        sout = derive_shape(dout, (ddata, din), (sdata, sin))
    nin, nout = np.prod(sin), np.prod(sout)

    # consistency check
    assert len(sdata) == len(ddata)
    assert len(sin) == len(din)
    assert len(sout) == len(dout)

    # data is scalar: use broadcasting
    if len(sdata) == 0:
        assert len(ddata) == 0
        if len(din) == 0:
            if len(dout) == 0:
                # case ",->" is trivial
                return sp.coo_matrix(([data],([0],[0])), shape=(1,1))
            else:
                data = data*np.ones(sout[0])
                ddata = dout[0]
                sdata = (sout[0],)
        else:
            data = data*np.ones(sin[0])
            ddata = din[0]
            sdata = (sin[0],)

    # broadcast a dimension for output if necessary
    d = ddata[0]
    iout = dout.find(d)
    if iout >= 0:
        assert sout[iout] == sdata[0]
        T_out = transposeop(np.prod(sout[iout:]), np.prod(sout[:iout]))
        dout = dout[iout:] + dout[:iout]
        sout = sout[iout:] + sout[:iout]
    else:
        T_out = sp.kron(np.ones((1,sdata[0])), sp.eye(nout))
        dout = ddata[0] + dout
        sout = (sdata[0],) + sout
    assert sout[0] == sdata[0]

    # strip left-most index from `data` and apply logic recursively
    iin = din.find(d)
    if iin >= 0:
        # case "j...,j...->j..."
        assert sin[iin] == sdata[0]
        T_in = transposeop(np.prod(sin[:iin]), np.prod(sin[iin:]))
        din, sin = din[iin:] + din[:iin], sin[iin:] + sin[:iin]
        subscripts = "%s,%s->%s" % (ddata[1:], din[1:], dout[1:])
        sin, sout = sin[1:], sout[1:]
        op = sp.block_diag([einsumop(subscripts, A, sin, sout) for A in data])
    else:
        # case "j...,...->j..."
        T_in = sp.eye(nin)
        subscripts = "%s,%s->%s" % (ddata[1:], din, dout[1:])
        sout = sout[1:]
        op = sp.vstack([einsumop(subscripts, A, sin, sout) for A in data])

    return T_out.dot(op).dot(T_in)
