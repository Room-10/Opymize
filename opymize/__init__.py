
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg

class Variable(object):
    def __init__(self, *args):
        if len(args) == 1 and type(args[0]) in [np.int64,int]:
            args = [(args[0],)]
        if type(args[0][0]) is not str:
            args = list([('v%d'%i, a) for i,a in enumerate(args)])
        self._descr = {}
        self._args = args
        self.size = 0
        for name, dim in args:
            self._descr[name] = {
                'name': name,
                'offset': self.size,
                'shape': dim,
                'size': np.prod(dim)
            }
            self.size += np.prod(dim)

    def vars(self, data=None, named=False):
        if data is None:
            if named:
                return self._descr
            else:
                return [self._descr[a[0]] for a in self._args]

        result = {} if named else [None for a in self._args]
        for i,a in enumerate(self._args):
            d = self._descr[a[0]]
            start, end = d['offset'], d['offset']+np.prod(d['shape'])
            idx = d['name'] if named else i
            result[idx] = data[start:end].reshape(d['shape'])
        return result

    def __getitem__(self, idx):
        if type(idx) is str:
            return self._descr[idx]
        else:
            return self._descr[self._args[idx][0]]

    def new(self, dtype=np.float64):
        return np.zeros((self.size,), order='C', dtype=dtype)

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
        if hasattr(self, 'spmat'):
            mat = self.spmat
        elif hasattr(self, 'mat'):
            mat = self.mat
        else:
            raise Exception("This operator does not support numpy arrays")

        assert y is not None
        x, y = x.ravel(), y.ravel()
        if not add:
            y[:] = mat.dot(x)
        else:
            y += mat.dot(x)

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
        if not add:
            y.fill(0.0)
        if hasattr(self, 'spmat'):
            y += sp.linalg.norm(self.spmat, ord=p, axis=1)
        elif hasattr(self, 'mat'):
            y += np.linalg.norm(self.mat, ord=p, axis=1)
        else:
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
