
from __future__ import division

import numpy as np

from pycuda import compiler, gpuarray
import pycuda.autoinit
import pycuda.driver

import logging, warnings
# ignore nvcc deprecation warnings
warnings.simplefilter('ignore', UserWarning)

def gpu_info():
    # print information about the current GPU
    attrs = pycuda.autoinit.device.get_attributes()
    devattr = pycuda.driver.device_attribute
    blockdims = (attrs[devattr.MAX_BLOCK_DIM_X],
                 attrs[devattr.MAX_BLOCK_DIM_Y],
                 attrs[devattr.MAX_BLOCK_DIM_Z])
    griddims = (attrs[devattr.MAX_GRID_DIM_X],
                attrs[devattr.MAX_GRID_DIM_Y],
                attrs[devattr.MAX_GRID_DIM_Z])
    logging.debug("MAX_BLOCK_DIM={blockdims}, " \
        "MAX_GRID_DIM={griddims}, MAX_THREADS_PER_BLOCK={maxthreads}, " \
        "TOTAL_CONSTANT_MEMORY={const_mem}, " \
        "MAX_SHARED_MEMORY_PER_BLOCK={shared_mem}, " \
        "MAX_REGISTERS_PER_BLOCK={registers}, " \
        "KERNEL_EXEC_TIMEOUT={kernel_timeout}".format(
        blockdims="x".join(map(str,blockdims)),
        griddims="x".join(map(str,griddims)),
        maxthreads=attrs[devattr.MAX_THREADS_PER_BLOCK],
        const_mem=attrs[devattr.TOTAL_CONSTANT_MEMORY],
        shared_mem=attrs[devattr.MAX_SHARED_MEMORY_PER_BLOCK],
        registers=attrs[devattr.MAX_REGISTERS_PER_BLOCK],
        kernel_timeout=attrs[devattr.KERNEL_EXEC_TIMEOUT],
    ))

def prepare_kernels(files, templates, constvars, blockvars={}):
    """ Compile and prepare CUDA kernel functions

    Args:
        files : list of cuda source file handles
        templates : list of tuples describing the kernels
        constvars : dict of readonly variables
        blockvars : dict of blockvars whose description will be included
                    in the preamble as preprocessor macros
    Returns:
        kernels : dict of executable CUDA kernels
    """
    preamble, constvars = prepare_vars(constvars, blockvars)
    kernels_code = preamble
    for f in files:
        kernels_code += f.read().decode("utf-8")

    mod = compiler.SourceModule(kernels_code)
    kernels = {}
    for d in templates:
        kernels[d[0]] = prepare_kernelfun(mod, *d)

    for name, val in constvars.items():
        const_ptr, size_in_bytes = mod.get_global(name)
        pycuda.driver.memcpy_htod(const_ptr, val)
        # WARNING: The gpudata argument in gpuarray.GPUArray usually requires a
        # pycuda.driver.DeviceAllocation and const_ptr is an int generated from
        # casting a CUdeviceptr to an int.
        # However, since DeviceAllocation is a simple wrapper around CUdeviceptr
        # (that gives a CUdeviceptr when cast to an int), it works like this.
        constvars[name] = gpuarray.GPUArray(val.shape, val.dtype, gpudata=const_ptr)

    return kernels

def prepare_kernelfun(mod, name, signature, totaldim, blockdim):
    griddim = tuple(int(np.ceil(t/b)) for t,b in zip(totaldim, blockdim))
    func = mod.get_function(name)
    func.prepare(signature)
    def kernelfun(*args):
        gpuargs = [a.gpudata if t == "P" else a for t,a in zip(signature,args)]
        return func.prepared_call(griddim, blockdim, *gpuargs)
    return kernelfun

def prepare_vars(constvars, blockvars):
    preamble = ""
    new_constvars = {}

    for name, val in constvars.items():
        if type(val) is str:
            if len(val) == 1:
                preamble += "#define %s ('%s')\n" % (name, val)
            else:
                preamble += "#define %s %s\n" % (name, val)
        elif type(val) is int or type(val) is np.int64:
            preamble += "#define %s (%d)\n" % (name, val)
        elif type(val) is bool:
            preamble += "#define %s (%d)\n" % (name, 1 if val else 0)
        elif type(val) is float or type(val) is np.float64:
            preamble += "__constant__ double %s = %s;\n" % (name, repr(val))
        elif type(val) is np.ndarray:
            if val.dtype == 'int64':
                preamble += "__constant__ int %s[%d] = { %s };\n" % (
                    name, val.size,
                    ", ".join(str(i) for i in val.ravel()))
            elif val.dtype == 'bool':
                preamble += "__constant__ unsigned char %s[%d] = { %s };\n" % (
                    name, val.size,
                    ", ".join(str(int(i)) for i in val.ravel()))
            elif val.dtype == 'float64':
                preamble += "__device__ double %s[%d];\n" % (name, val.size)
                new_constvars[name] = val
            else:
                raise Exception("Numpy dtype not supported: %s" % val.dtype)
        else:
            raise Exception("Type of '%s' not supported: %s" % (name, type(val)))

    for name, val in blockvars.items():
        # Encode BlockVar description as preprocessor macros
        for subvar in val:
            subname = subvar['name'] + name[1:]
            preamble += "#define SUBVAR_%s_%s(X,Y) double * X = & Y [%d];\n" \
                            % (name, subname, subvar['offset'])

    return preamble, new_constvars
