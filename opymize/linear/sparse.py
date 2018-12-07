
import warnings

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
            one of 'neumann' (default), 'dirichlet', 'second-order' or 'curvature'
    """
    if steps is None:
        steps = np.ones(len(dims))

    def dd(n):
        diags = np.ones(n)*np.array([[1, -2, 1]]).T
        if boundaries in ["curvature", "second-order"]:
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
    return einsumop("ij,jk->ik", Delta, dims={ 'k': components })

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
        partial_t = extendedop(partial_t, before=dims[:t], after=dims[(t+1):])
        partials.append(partial_t)
    return sp.kron(einsumop("tij,j->it", partials), sp.diags(weights))

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

        diags = [0.5*d for d in diags]
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
            A = extendedop(A, before=dims[:ti], after=dims[ti+1:])
            avg_t = 0.5*(avg_t + A)
        avgs.append(avg_t)
    return einsumop("tij,jt->it", avgs)

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
    idx = strides.dot(np.indices(shape, dtype=np.int64).reshape(ndims, -1))
    data = (np.ones(N, dtype=np.int8), (np.arange(N), idx))
    return sp.coo_matrix(data, shape=(N,N), dtype=np.int8)

def transposeop(m, n):
    return transposeopn((m,n), (1,0))

def idxop(P, m):
    # y[i] = x[P[i]]  where  P[i] \in {0,...,m}
    return sp.coo_matrix((np.ones(P.size),(range(P.size),P)), shape=(P.size,m))

def extendedop(A, before=(), after=()):
    """ y[...,i,...] = sum_j A[i,j]*x[...,j,...] """
    dims = { 'k': int(np.prod(before)), 'l': int(np.prod(after)) }
    return einsumop("ij,kjl->kil", A, dims=dims)

def block_diag_csr(blocks):
    heights, widths = [[B.shape[i] for B in blocks] for i in [0,1]]
    indptr = np.r_[[0], np.cumsum(np.repeat(widths, heights))]
    indices = np.r_[[0], np.cumsum(widths[:-1])]
    indices = map(np.add, indices, map(np.arange, widths))
    indices = np.hstack(map(np.tile, indices, heights))
    data = np.hstack([B.ravel() for B in blocks])
    return sp.csr_matrix((data, indices, indptr))

def array_shape(data):
    if hasattr(data, 'shape'):
        return sp.issparse(data), data.shape
    elif type(data) in [list,tuple] and len(data) > 0:
        issparse, subshape = array_shape(data[0])
        return issparse, (len(data),) + subshape
    elif np.isscalar(data):
        return False, ()
    else:
        raise ValueError("Unknown shape: '%s'" % type(data))

def einsumop(subscripts, data, shape_in=None, shape_out=None, order='C', dims={}):
    """ Generate sparse operator from numpy.einsum subscripts

    Args:
        subscripts : string
            tensor dot string, see documention of numpy.einsum
        data : array-like object
            Can be a list of sparse matrices or a scalar (for 0-dimensional)
            or an ndarray or a list of ndarrays etc.
        shape_in : tuple of ints
            If None, the input shape is determined from `subscripts` and `data`.
        shape_out : tuple of ints
            If None, the output shape is determined from `subscripts`, `data`
            and `shape_in`.
        order : {'C','F'} or pair of these
            Assume that input/output have specified ordering.
        dims : dict
            Specify dimensions for letters in `subscripts` that can't be
            determined from given `data`, `shape_in` and `shape_out`.

    Examples:
        >>> subscripts = "ijkl,kij->lj"
        >>> # suppose `x`, `data` are arrays of correct shape
        >>> A = einsumop(subscripts, data)
        >>> A.dot(x.ravel()) == np.einsum(subscripts, data, x).ravel()
    """
    din, dout = subscripts.split("->")
    ddata, din = din.split(",")

    issparse, sdata = array_shape(data)
    data = np.asarray(data, dtype=object if issparse else None)

    # determine shapes of input/output vectors
    dims.update((d, sdata[i]) for i,d in enumerate(ddata))
    sin = [dims[d] for d in din] if shape_in is None else shape_in
    dims.update((d, sin[i]) for i,d in enumerate(din))
    sout = [dims[d] for d in dout] if shape_out is None else shape_out
    dims.update((d, sout[i]) for i,d in enumerate(dout))
    nin, nout = int(np.prod(sin)), int(np.prod(sout))

    # convert requested ordering to 'C' ordering
    if order in ['C','F']:
        order = (order, order)
    din = din[::-1] if order[0] == 'F' else din
    sin = sin[::-1] if order[0] == 'F' else sin
    dout = dout[::-1] if order[1] == 'F' else dout
    sout = sout[::-1] if order[1] == 'F' else sout

    # consistency check
    assert len(sdata) == len(ddata)
    assert len(sin) == len(din)
    assert len(sout) == len(dout)

    # eliminate indices that only appear in input/output vectors
    dcommon = "".join(d for d in din if d not in ddata and d in dout)

    ain1 = [i for i,d in enumerate(din) if d not in ddata+dout]
    ain2 = [din.index(d) for d in dcommon]
    ain3 = [i for i in range(len(din)) if i not in ain1+ain2]
    T_in = transposeopn(sin, ain2+ain3+ain1)
    if len(ain1) != 0:
        n1 = np.prod([sin[i] for i in ain1])
        T_in = sp.kron(sp.eye(nin/n1), np.ones((1,n1))).dot(T_in)

    aout1 = [i for i,d in enumerate(dout) if d not in ddata+din]
    aout2 = [dout.index(d) for d in dcommon]
    aout3 = [i for i in range(len(dout)) if i not in aout1+aout2]
    T_out = transposeopn(sout, aout1+aout2+aout3).T
    if len(aout1) != 0:
        n1 = np.prod([sout[i] for i in aout1])
        T_out = T_out.dot(sp.kron(np.ones((n1,1)), sp.eye(nout/n1)))

    extra = int(np.prod([sin[i] for i in ain2]))
    din = "".join(din[i] for i in ain3)
    sin = tuple(sin[i] for i in ain3)
    dout = "".join(dout[i] for i in aout3)
    sout = tuple(sout[i] for i in aout3)
    nin, nout = int(np.prod(sin)), int(np.prod(sout))

    dcommon = "".join(d for d in ddata if d in din and d in dout)
    if len(dcommon) > 0:
        # handle indices that appear on all operands by recursion and block_diag
        adata1 = [ddata.index(d) for d in dcommon]
        adata2 = [i for i in range(len(ddata)) if i not in adata1]
        if issparse and len(sdata)-1 in adata1 or len(sdata)-2 in adata1:
            warnings.warn("Conversion to dense array in einsumop", UserWarning)
            data = np.asarray([a.toarray() for a in data.ravel()])
            data = data.reshape(sdata)
            issparse = False

        if issparse:
            adata3 = [i for i in adata2 if i < len(sdata)-2]
            sdata3 = tuple(sdata[i] for i in adata3)
            sdata = sdata3 + sdata[-2:]
            ddata = "".join(ddata[i] for i in adata3) + ddata[-2:]
            data = data.transpose(adata1+adata3).reshape((-1,) + sdata3)
        else:
            sdata = tuple(sdata[i] for i in adata2)
            ddata = "".join(ddata[i] for i in adata2)
            data = data.transpose(adata1+adata2).reshape((-1,) + sdata)

        ain1 = [din.index(d) for d in dcommon]
        ain2 = [i for i in range(len(din)) if i not in ain1]
        T_in2 = transposeopn(sin, ain1+ain2)
        din = "".join(din[i] for i in ain2)
        sin = tuple(sin[i] for i in ain2)

        aout1 = [dout.index(d) for d in dcommon]
        aout2 = [i for i in range(len(dout)) if i not in aout1]
        T_out2 = transposeopn(sout, aout1+aout2).T
        dout = "".join(dout[i] for i in aout2)
        sout = tuple(sout[i] for i in aout2)

        subscripts = "%s,%s->%s" % (ddata, din, dout)
        op = sp.block_diag([einsumop(subscripts, A, sin, sout) for A in data])
        op = T_out2.dot(op).dot(T_in2)
    elif issparse and len(ddata) > 2 and all(d in dout for d in ddata[:-2]):
        aout1 = [dout.index(d) for d in ddata[:-2]]
        aout2 = [i for i in range(len(dout)) if i not in aout1]
        T_out2 = transposeopn(sout, aout1+aout2).T
        dout = "".join(dout[i] for i in aout2)
        subscripts = "%s,%s->%s" % (ddata[-2:], din, dout)
        op = sp.vstack([einsumop(subscripts, A, sin) for A in data.ravel()])
        op = T_out2.dot(op)
    elif issparse and len(ddata) == 2 and ddata in [din+dout, dout+din]:
        # only trivial cases are supported for sparse data
        data = data.ravel()[0]
        if ddata == din:
            # case "ij,ij->"
            op = data.reshape(1,-1)
        elif ddata == dout:
            # case "ij,->ij"
            op = data.reshape(-1,1)
        elif ddata[1] == din:
            # case "ij,j->i"
            op = data
        else:
            # case "ij,i->j"
            op = data.T
    else:
        if issparse:
            warnings.warn("Conversion to dense array in einsumop", UserWarning)
            data = np.asarray([a.toarray() for a in data.ravel()])
            data = data.reshape(sdata)
            issparse = False

        # sum over indices that appear only in data array
        sum_axes = tuple([i for i,d in enumerate(ddata) if d not in dout+din])
        data = data.sum(axis=sum_axes)

        # reshaping reduces to simple matrix vector multiplication
        data = data.transpose([ddata.index(d) for d in dout+din])
        op = data.reshape(int(np.prod(sout)), int(np.prod(sin)))

    return T_out.dot(sp.kron(sp.eye(extra), op)).dot(T_in)
