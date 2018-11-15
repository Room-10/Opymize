
import numpy as np

from opymize.linear import IndexedMultAdj, MatrixMultRBatched, TangledMatrixMultR
from opymize.tools.tests import test_adjoint, test_rowwise_lp, test_gpu_op

if __name__ == "__main__":
    L = 50
    N = 64*64
    M = 60
    S = 2

    A = np.random.randn(M, S, S)
    B = np.random.randn(M, S, S+1)
    P = np.random.randint(0, high=L, size=(M,S+1))
    tmpMat = np.random.randn(M, S+1, 1, 1)

    IMAOp = IndexedMultAdj(L, N, P, B)
    MMRBOp = MatrixMultRBatched(N, A)
    TMMROp = TangledMatrixMultR(N, tmpMat)

    for operator in [IMAOp, MMRBOp, TMMROp]:
        for op in [operator,operator.adjoint]:
            test_adjoint(op)
            test_rowwise_lp(op)
            test_gpu_op(op)