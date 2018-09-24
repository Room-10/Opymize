
import numpy as np

class Variable(object):
    def __init__(self, *args):
        if len(args) == 1 and type(args[0]) in [np.int64,int]:
            args = [(args[0],)]
        self._descr = []
        self._args = args
        self.size = 0
        for dim in args:
            self._descr.append({
                'offset': self.size,
                'shape': dim
            })
            self.size += np.prod(dim)

    def vars(self, data=None):
        if data is None:
            return self._descr

        result = []
        for d in self._descr:
            size = np.prod(d['shape'])
            result.append(data[d['offset']:d['offset']+size].reshape(d['shape']))
        return result

    def __getitem__(self, idx):
        return self._descr[idx]

    def new(self, dtype=np.float64):
        return np.zeros((self.size,), order='C', dtype=dtype)

class BlockVar(object):
    def __init__(self, *args):
        self._descr = {}
        self._args = args
        for a in args:
            self._append(*a)
        self.size = self._size() # size is fixed after init
        self.data = np.zeros((self.size,), order='C')

    def _append(self, name, dim):
        offset = self._size()
        self._descr[name] = (offset, dim)

    def _size(self):
        return sum([np.prod(d[1]) for d in self._descr.values()])

    def copy(self):
        cpy = BlockVar(*self._args)
        cpy.data[:] = self.data
        return cpy

    def vars(self):
        return [self[a[0]] for a in self._args]

    def __getitem__(self, idx):
        if isinstance(idx, str):
            offset, dim = self._descr[idx]
            size = np.prod(dim)
            return self.data[offset:offset+size].reshape(dim)
        else:
            return self.data[idx]

    def __setitem__(self, idx, value):
        if isinstance(idx, str):
            offset, dim = self._descr[idx]
            size = np.prod(dim)
            self.data[offset:offset+size] = value
        else:
            if isinstance(value, BlockVar):
                value = value.data
            self.data[idx] = value

    def __sub__(self, other):
        return self + (-1.0)*other

    def __add__(self, other):
        if isinstance(other, BlockVar):
            other = other.data
        out = self.copy()
        out.data[:] += other
        return out

    def __rmul__(self, scalar):
        out = self.copy()
        out.data[:] *= scalar
        return out

    def __mul__(self, scalar):
        return scalar*self

    def __str__(self):
        return self.data.__str__()

    def __iter__(self):
        for d in self._args:
            yield { 'name': d[0], 'offset': self._descr[d[0]][0] }

class Operator(object):
    """ Representation of a mathematical operator T """
    def __init__(self):
        self.x = None # Variable, input (domain)
        self.y = None # Variable, output (range)

    def prepare_gpu(self, type_t="double"):
        # prepare/compile gpu kernels if necessary
        pass

    def _call_gpu(self, x, y=None, add=False, jacobian=False):
        raise Exception("This operator does not support GPU arrays")

    def _call_cpu(self, x, y=None, add=False, jacobian=False):
        raise Exception("This operator does not support numpy arrays")

    def __call__(self, x, y=None, add=False, jacobian=False):
        """ Evaluate T(x)

        The result is written to x by default.

        Args:
            x : array (either numpy or gpuarray)
            y : if specified, the result is written to this array
            add : if True, add the result to the target array's current value
            jacobian : if True, return the jacobian linear operator of T

        Returns:
            The jacobian of T as LinOp if jacobian==True, else None.
        """
        try:
            from pycuda import gpuarray
            if type(x) is gpuarray.GPUArray:
                return self._call_gpu(x, y=y, add=add, jacobian=jacobian)
        except ImportError:
            pass
        return self._call_cpu(x, y=y, add=add, jacobian=jacobian)

class LinOp(Operator):
    """ (Matrix-free) representation of a linear operator A """
    def __init__(self):
        Operator.__init__(self)
        self.adjoint = None # another LinOp

    def _call_gpu(self, x, y=None, add=False):
        raise Exception("This operator does not support GPU arrays")

    def _call_cpu(self, x, y=None, add=False):
        raise Exception("This operator does not support numpy arrays")

    def __call__(self, x, y=None, add=False, jacobian=False):
        try:
            from pycuda import gpuarray
            if type(x) is gpuarray.GPUArray:
                self._call_gpu(x, y=y, add=add)
                return self if jacobian else None
        except ImportError:
            pass
        self._call_cpu(x, y=y, add=add)
        # a linear operator is its own jacobian!
        return self if jacobian else None

    def rowwise_lp(self, y, p=1, add=False):
        """ Compute the rowwise l^p norm of A

        y_i = |A_i|_p^p = \sum_j |A_ij|^p

        Args:
            y : numpy array of shape (A.y.size,)
            p : positive float
            add : if True, add the result to the current value of y

        Returns:
            Nothing, the result is written to y.
        """
        raise Exception("This operator does not support rowwise l^p norms.")

class Functional(object):
    """ Representation of a mathematical functional F """
    def __init__(self):
        self.x = None # Variable, input (domain)
        self.conj = None # Functional, convex conjugate

    def __call__(self, x, grad=False):
        """ Evaluate F(x)

        Args:
            x : numpy array
            grad : if True, return the gradient of F at x

        Returns:
            (val,infeas) or ((val,infeas), grad) if grad==True
            val : the value of
            infeas : for functionals taking the value infty, this is a measure
                for the distance to the feasible set
            grad : same shape as x
        """
        raise Exception("This functional does not support being called")

    def prox(self, tau):
        """ Proximal operator of F

        In general:
            F.prox(tau)(x1) = argmin(x2)[0.5*|x2-x1|^2 + tau*F(x2)]
            x = F.prox(tau)(x) + tau*F.conj.prox(1/tau)(x/tau)

        Args:
            tau : scalar or numpy array of shape (self.x.size,)

        Returns:
            Instance of opymize.operators.Operator
        """
        raise Exception("This functional does not support taking prox")
