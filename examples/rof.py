
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

def main(lbd=40.0, input_file=None, new_m=None):
    if input_file is None:
        from skimage.data import astronaut
        orig_data = np.array(astronaut(), dtype=np.float64)
        orig_data += np.sqrt(lbd)*np.random.randn(*orig_data.shape)
        output_file = "astronaut-%.1f.png" % (lbd,)
    else:
        input_base, _, input_ext = input_file.rpartition(".")
        output_file = "%s-%.1f.%s" % (input_base, lbd, input_ext)
        orig_data = imread(input_file)

    m = np.array(orig_data.shape[:-1], dtype=np.int64)
    logging.info("Image size: %dx%d" % (m[0], m[1]))

    if new_m is not None:
        new_m = np.array(new_m, dtype=np.int64).ravel()
        assert(new_m.shape == (2,))
        logging.info("Goal: Embed into %dx%d using ROF (lbd=%.1f) inpainting" \
                     % (new_m[0], new_m[1], lbd))
        pad = (new_m - m)//2
        data = np.zeros((new_m[0],new_m[1],3), order='C', dtype=np.float64)
        data[pad[0]:pad[0]+m[0],pad[1]:pad[1]+m[1],:] = orig_data
        mask = np.zeros(new_m, order='C', dtype=bool)
        mask[pad[0]:pad[0]+m[0],pad[1]:pad[1]+m[1]] = True
        mask = mask.ravel()
    else:
        logging.info("Goal: Denoise using ROF (lbd=%.1f)" % (lbd,))
        data = np.array(orig_data, dtype=np.float64)
        mask = None

    imagedims = data.shape[:-1]
    n_image = np.prod(imagedims)
    d_image = len(imagedims)
    l_labels = data.shape[-1]

    G = SSD(data.reshape(-1, l_labels), mask=mask)
    # alternatively constrain to the input data:
    #G = ConstrainFct(mask, data.reshape(-1, l_labels))
    F = L1Norms(n_image, (l_labels, d_image), lbd=lbd)
    linop = GradientOp(imagedims, l_labels)

    solver = PDHG(G, F, linop)
    solver.solve(steps="precond", precision="double", use_gpu=True,
                 term_pd_gap=1e-5, term_maxiter=int(1e4), granularity=int(1e3))

    ## testing a new semismooth (quasi-)newton solver:
    #pdhg_state = solver.state
    #result = np.concatenate(pdhg_state)
    #from opymize.solvers import SemismoothQuasinewton
    #solver = SemismoothQuasinewton(G, F, linop)
    #solver.solve(term_relgap=1e-10, continue_at=result)
    #
    ## for comparison: continue PDHG
    #solver = PDHG(G, F, linop)
    #solver.solve(steps='precond', term_maxiter=5000, granularity=500,
    #             term_relgap=1e-10, use_gpu=True, continue_at=pdhg_state)

    result = solver.state[0].reshape(data.shape)
    result = np.asarray(np.clip(result, 0, 255), dtype=np.uint8)
    if l_labels == 1:
        result = result[:,:,0]
    logging.info("Writing result to '%s'..." % output_file)
    imsave(output_file, result)

if __name__ == "__main__":
    lbd = 100.0
    input_file = None
    if len(sys.argv) > 1:
        input_file = np.float64(sys.argv[1])
        if len(sys.argv) > 2:
            lbd = np.float64(sys.argv[2])
    main(lbd=lbd, input_file=input_file)
