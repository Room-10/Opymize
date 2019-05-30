
import numpy as np

def truncate(x, n):
    k = -int(np.floor(np.log10(abs(x))))
    # Example: x = 0.006142 => k = 3 / x = 2341.2 => k = -3
    k += n - 1
    if k > 0:
        x_str = str(abs(x))[:(k+2)]
    else:
        x_str = str(abs(x))[:n]+"0"*(-k)
    return np.sign(x)*float(x_str)

def solve_reduced_monic_cubic(a, b, soln=0):
    """ Solve x**3 + a*x + b = 0 using explicit formulas.

    Only real solutions are computed and in case more than one real
    solution exists, only one of them is returned.

    Args:
        a, b : array-like of floats, shape (nvals,)
        soln : int, one of 0,1,2
            Indicate which solution is returned in case of non-uniqueness.

    Returns:
        x : ndarray of floats, shape (nvals,)
    """
    a, b = np.asarray(a), np.asarray(b)
    assert a.size == b.size
    a, b = a.ravel(), b.ravel()
    x, Q, Q3, R, D, arr1, arr2 = [np.zeros_like(a) for i in range(7)]

    # trivial case (a == 0):
    msk = (a == 0)
    x[msk] = np.cbrt(-b[msk])

    # nontrivial case (a != 0):
    msk = ~msk
    Q[msk], R[msk] = a[msk]/3, -b[msk]/2
    Q3[msk] = Q[msk]**3
    D[msk] = Q3[msk] + R[msk]**2

    # subcase with three real roots:
    msk2 = msk & (D <= 0)
    theta, sqrt_Q = arr1, arr2
    theta[msk2] = np.arccos(R[msk2]/np.sqrt(-Q3[msk2]))
    sqrt_Q[msk2] = np.sqrt(-Q[msk2])
    x[msk2] = 2*sqrt_Q[msk2]*np.cos((theta[msk2] + 2*soln*np.pi)/3.0)

    # subcase with unique real root:
    msk2 = msk & (D > 0)
    AD, BD = arr1, arr2
    AD[msk2] = np.cbrt(np.abs(R[msk2]) + np.sqrt(D[msk2]))*np.sign(R[msk2])
    msk3 = msk2 & (AD != 0)
    BD[msk3] = -Q[msk3]/AD[msk3]
    x[msk2] = AD[msk2] + BD[msk2]

    return x
