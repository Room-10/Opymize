
from opymize import Variable, Operator
from opymize.linear.scale import ScaleOp, ZeroOp

import numpy as np

try:
    import opymize.tools.gpu
    from pycuda import gpuarray
    from pycuda.elementwise import ElementwiseKernel
except:
    # no cuda support
    pass

class ConstOp(Operator):
    """ T(x) = const, independent of x """
    def __init__(self, N, const):
        Operator.__init__(self)
        self.x = Variable(N)
        self.y = Variable(const.size)
        self.const = const
        self._jacobian = ZeroOp(self.x.size, self.y.size)

    def prepare_gpu(self):
        self.gpu_const = gpuarray.to_gpu(self.const)
        self._jacobian.prepare_gpu()

    def _call_gpu(self, x, y=None, add=False, jacobian=False):
        y = x if y is None else y
        if add:
            y += self.gpu_const
        else:
            y[:] = self.gpu_const
        if jacobian:
            return self._jacobian

    def _call_cpu(self, x, y=None, add=False, jacobian=False):
        y = x if y is None else y
        if add:
            y += self.const
        else:
            y[:] = self.const
        if jacobian:
            return self._jacobian

class ConstrainOp(Operator):
    """ y[mask,:] = const, else identity """
    def __init__(self, mask, const):
        Operator.__init__(self)
        self.x = Variable(const.shape)
        self.y = self.x
        self.mask = mask
        self.const = const
        scale = np.ones_like(const)
        scale[mask,:] = 0.0
        self._jacobian = ScaleOp(self.x.size, scale.ravel())

    def prepare_gpu(self):
        self.const_gpu = gpuarray.to_gpu(self.const)
        self.mask_gpu = gpuarray.to_gpu(self.mask.astype(np.int8))
        N, M = self.x[0]['shape']
        headstr = "double *x, double *y, double *c, char *mask"
        self._kernel = ElementwiseKernel(headstr,
            "y[i] = (mask[i/{}]) ? c[i] : x[i]".format(M))
        self._kernel_add = ElementwiseKernel(headstr,
            "y[i] += (mask[i/{}]) ? c[i] : x[i]".format(M))
        self._jacobian.prepare_gpu()

    def _call_gpu(self, x, y=None, add=False, jacobian=False):
        y = x if y is None else y
        if add: self._kernel_add(x, y, self.const_gpu, self.mask_gpu)
        else: self._kernel(x, y, self.const_gpu, self.mask_gpu)
        if jacobian:
            return self._jacobian

    def _call_cpu(self, x, y=None, add=False, jacobian=False):
        if y is not None:
            y[:] = x
            x = y
        x = self.x.vars(x)[0]
        x[self.mask,:] = self.const[self.mask,:]
        if jacobian:
            return self._jacobian

class ShiftScaleOp(Operator):
    """ T(x) = a*(x + b*shift), where a, b, shift can be float or arrays """
    def __init__(self, N, shift, a, b):
        Operator.__init__(self)
        self.shift = shift
        self.a = a
        self.b = b
        self.x = Variable(N)
        self.y = Variable(N)
        self._jacobian = ScaleOp(N, self.a)

    def prepare_gpu(self):
        # don't multiply with a if a is 1 (not 1.0!)
        afact = "" if self.a is 1 else "a[0]*"
        if type(self.a) is np.ndarray:
            afact = "a[i]*"

        # don't multiply with b if b is 1 (not 1.0!)
        bfact = "" if self.b is 1 else "b[0]*"
        if type(self.b) is np.ndarray:
            bfact = "b[i]*"

        # don't shift if shift is 0 (not 0.0!)
        shiftstr = " + %sshift" % bfact
        shiftstr += "[i]" if type(self.shift) is np.ndarray else "[0]"
        if self.shift is 0 or self.b is 0:
            shiftstr = ""

        self.gpuvars = {
            'shift':    gpuarray.to_gpu(np.array(self.shift, dtype=np.float64, ndmin=1)),
            'a':        gpuarray.to_gpu(np.array(self.a, dtype=np.float64, ndmin=1)),
            'b':        gpuarray.to_gpu(np.array(self.b, dtype=np.float64, ndmin=1))
        }
        headstr = "double *x, double *y, double *shift, double *a, double *b"
        self._kernel = ElementwiseKernel(headstr,
            "y[i] = %s(x[i]%s)" % (afact, shiftstr))
        self._kernel_add = ElementwiseKernel(headstr,
            "y[i] += %s(x[i]%s)" % (afact, shiftstr))

        self._jacobian.prepare_gpu()

    def _call_gpu(self, x, y=None, add=False, jacobian=False):
        g = self.gpuvars
        y = x if y is None else y
        krnl = self._kernel_add if add else self._kernel
        krnl(x, y, g['shift'], g['a'], g['b'])

        if jacobian:
            self._jacobian.fact = self.a
            return self._jacobian

    def _call_cpu(self, x, y=None, add=False, jacobian=False):
        y = x if y is None else y
        if add:
            y += self.a*(x + self.b*self.shift)
        else:
            y[:] = self.a*(x + self.b*self.shift)

        if jacobian:
            self._jacobian.fact = self.a
            return self._jacobian
