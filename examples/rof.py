
"""
    Implementation of the Rudin-Osher-Fatemi (L2-TV) image restoration model
    for color (RGB) input images.
"""

# Pretty log output
import sys, logging
class MyFormatter(logging.Formatter):
    def format(self, record):
        th, rem = divmod(record.relativeCreated/1000.0, 3600)
        tm, ts = divmod(rem, 60)
        record.relStrCreated = "% 2d:%02d:%06.3f" % (int(th),int(tm),ts)
        return super(MyFormatter, self).format(record)
ch = logging.StreamHandler(sys.stdout)
ch.setLevel(logging.INFO)
ch.setFormatter(MyFormatter('[%(relStrCreated)s] %(message)s'))
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
logger.addHandler(ch)

from opymize.solvers import PDHG
from opymize.functionals import ConstrainFct, SSD, L1Norms
from opymize.linear.diff import GradientOp

import numpy as np

try:
    from skimage.io import imread, imsave
except ImportError:
    print("This example requires `scikit-image` to run!")
    sys.exit()

def main():
    input_file = "noisy.png"
    output_file = "out.png"

    orig_data = imread(input_file)
    m = np.array(orig_data.shape[:-1], dtype=np.int64)
    logging.info("Original size: %dx%d" % (m[0], m[1]))

    new_m = np.array([400, 400], dtype=np.int64)
    lbd = 40
    logging.info("Goal: Embed into %dx%d using ROF (lbd=%.1f) inpainting" \
                 % (new_m[0], new_m[1], lbd))

    padding = (new_m - m)//2
    data = np.zeros((new_m[0],new_m[1],3), order='C', dtype=np.float64)
    data[padding[0]:padding[0]+m[0],padding[1]:padding[1]+m[1],:] = orig_data

    imagedims = data.shape[:-1]
    n_image = np.prod(imagedims)
    d_image = len(imagedims)
    l_labels = data.shape[-1]

    mask = np.zeros(imagedims, order='C', dtype=bool)
    mask[padding[0]:padding[0]+m[0],padding[1]:padding[1]+m[1]] = True
    mask = mask.ravel()

    G = SSD(data.reshape(-1, l_labels), mask=mask)
    # alternatively constrain to the input data:
    #G = ConstrainFct(mask, data.reshape(-1, l_labels))
    F = L1Norms(n_image, (l_labels, d_image), lbd=lbd)
    linop = GradientOp(imagedims, l_labels)

    solver = PDHG(G, F, linop)
    solver.solve(steps='precond', term_maxiter=5000, granularity=500, use_gpu=True)

    ## testing a new semismooth newton solver:
    #result = np.concatenate(solver.state)
    #from opymize.solvers import SemismoothNewton
    #solver = SemismoothNewton(G, F, linop)
    #solver.solve(continue_at=result)

    result = solver.state[0].reshape(data.shape)
    result = np.asarray(np.clip(result, 0, 255), dtype=np.uint8)
    if l_labels == 1:
        result = result[:,:,0]
    logging.info("Writing result to '%s'..." % output_file)
    imsave(output_file, result)

if __name__ == "__main__":
    main()