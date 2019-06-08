
#ifdef L1_NORMS_PROJ
__global__ void l1normsproj(TYPE_T *x)
{
    /* This function makes heavy use of registers (34 32-bit registers), so
     * that it will not run with more than 960 threads per block on compute
     * capability 2.x!
     *
     * x_i = proj(x_i, lbd)
     */
#if (M1 <= M2)
// A := x_i, a (M1 x M2)-matrix
#define LIM M2
#define STEP1 M1
#define STEP2 (1)
#else
// A := x_i^T, a (M2 x M1)-matrix
#define LIM M1
#define STEP1 (1)
#define STEP2 M1
#endif

    // global thread index
    int i = blockIdx.x*blockDim.x + threadIdx.x;

    // stay inside maximum dimensions
    if (i >= N) return;

    // iteration variables and misc.
    int mm;
    TYPE_T *xi = &x[i*(M1*M2)];
    TYPE_T norm = 0.0;

#if (M1 == 1 || M2 == 1 || matrixnorm == 'F')
    for (mm = 0; mm < M1*M2; mm++) {
        norm += xi[mm]*xi[mm];
    }

    if (norm > lbd*lbd) {
        norm = lbd/SQRT(norm);
        for (mm = 0; mm < M1*M2; mm++) {
            xi[mm] *= norm;
        }
    }
#elif (M1 == 2 || M2 == 2)
    TYPE_T C11 = 0.0, C12 = 0.0, C22 = 0.0,
           V11 = 0.0, V12 = 0.0, V21 = 0.0, V22 = 0.0,
           M11 = 0.0, M12 = 0.0, M21 = 0.0, M22 = 0.0,
           s1 = 0.0, s2 = 0.0,
           trace, d, lmax, lmin, smax, smin;

    // C = A^T A, a (2 x 2)-matrix
    for (mm = 0; mm < LIM; mm++) {
        C11 += xi[mm*STEP1 + 0*STEP2]*xi[mm*STEP1 + 0*STEP2];
        C12 += xi[mm*STEP1 + 0*STEP2]*xi[mm*STEP1 + 1*STEP2];
        C22 += xi[mm*STEP1 + 1*STEP2]*xi[mm*STEP1 + 1*STEP2];
    }

    // Compute eigenvalues
    trace = C11 + C22;
    d = SQRT(FMAX(0.0, 0.25*trace*trace - (C11*C22 - C12*C12)));
    lmax = FMAX(0.0, 0.5*trace + d);
    lmin = FMAX(0.0, 0.5*trace - d);
    smax = SQRT(lmax);
    smin = SQRT(lmin);

    if (smax > lbd) {
        // Compute orthonormal eigenvectors
        if (C12 == 0.0) {
            if (C11 >= C22) {
                V11 = 1.0; V12 = 0.0;
                V21 = 0.0; V22 = 1.0;
            } else {
                V11 = 0.0; V12 = 1.0;
                V21 = 1.0; V22 = 0.0;
            }
        } else {
            V11 = C12       ; V12 = C12;
            V21 = lmax - C11; V22 = lmin - C11;
            norm = HYPOT(V11, V21);
            V11 /= norm; V21 /= norm;
            norm = HYPOT(V12, V22);
            V12 /= norm; V22 /= norm;
        }

        // Thresholding of eigenvalues
        s1 = FMIN(smax, lbd)/smax;
        s2 = FMIN(smin, lbd);
        s2 = (smin > 0.0) ? s2/smin : 0.0;

        // M = V * diag(s) * V^T
        M11 = s1*V11*V11 + s2*V12*V12;
        M12 = s1*V11*V21 + s2*V12*V22;
        M21 = s1*V21*V11 + s2*V22*V12;
        M22 = s1*V21*V21 + s2*V22*V22;

        // proj(A) = A * M
        for (mm = 0; mm < LIM; mm++) {
            // s1, s2 now used as temp. variables
            s1 = xi[mm*STEP1 + 0*STEP2];
            s2 = xi[mm*STEP1 + 1*STEP2];
            xi[mm*STEP1 + 0*STEP2] = s1*M11 + s2*M21;
            xi[mm*STEP1 + 1*STEP2] = s1*M12 + s2*M22;
        }
    }
#endif
}
#endif

#ifdef QUAD_EPI_PROJ
inline __device__ TYPE_T solve_reduced_monic_cubic(TYPE_T a, TYPE_T b)
{
    /* Solve x**3 + a*x + b = 0 using explicit formulas.
     *
     * Only real solutions are computed and in case more than one real
     * solution exists, only one of them is returned.
     **/
    if (a == 0.0) {
        return (b == 0.0) ? 0.0 : CBRT(-b);
    } else if (b == 0.0 && a > 0.0) {
        return 0.0;
    }

    TYPE_T theta, sqrt_Q, AD;
    TYPE_T Q = a/3.0;
    TYPE_T R = -b/2.0;
    TYPE_T Q3 = Q*Q*Q;
    TYPE_T D = Q3 + R*R;

    if (D <= 0.0) {
        theta = ACOS(R/SQRT(-Q3));
        sqrt_Q = SQRT(-Q);
        return 2.0*sqrt_Q*COS(theta/3.0);
    } else {
        AD = CBRT(FABS(R) + SQRT(D));
        if (R < 0.0) AD *= -1.0;
        return AD - Q/AD;
    }
}

__global__ void quadepiproj(TYPE_T *x)
{
    /* Project (x1,x2) onto the epigraph of a paraboloid
     *
     *      f(x1) := 0.5*alph*|x1|^2 \leq x2,
     *
     * optionally shifted and truncated at |x1 - shift1| < lbd.
     *
     * Note that the multi-dimensional case can be reduced to the scalar case
     * by radial symmetry.
     *
     * After translation, the projection (y1,y2) satisfies the orth. condition
     *
     *      (|x1| - |y1|) + (x2 - 0.5*a*|y1|^2)*(a*|y1|) = 0,
     *
     * which is a reduced monic cubic equation in |y1|.
     *
     * The result (projection) is stored in place.
     **/

    // global thread index
    int i = blockIdx.x*blockDim.x + threadIdx.x;

    // stay inside maximum dimensions
    if (i >= N) return;

    // iteration variables and misc.
    int mm;
    TYPE_T *x1 = &x[i*(M+1)];
    TYPE_T *x2 = &x[i*(M+1) + M];
    TYPE_T x1norm, y1norm, l;
    TYPE_T x1norm_sq = 0.0;

#ifdef USE_SHIFT
    for (mm = 0; mm < M+1; mm++) {
        x[i*(M+1) + mm] -= shift[i*(M+1) + mm];
    }
#endif

    for (mm = 0; mm < M; mm++) {
        x1norm_sq += x1[mm]*x1[mm];
    }

    if (x1norm_sq > 0) {
        x1norm = SQRT(x1norm_sq);
#ifdef USE_LBD
        l = -x1norm/(lbd*alph) + 0.5*alph*lbd*lbd + 1.0/alph;
        if (l <= x2[0]) {
            if (x1norm > lbd) {
                x1norm = lbd/x1norm;
                for (mm = 0; mm < M; mm++) {
                    x1[mm] *= x1norm;
                }
                l = 0.5*alph*lbd*lbd;
                if (l > x2[0]) {
                    x2[0] = l;
                }
            }
        } else
#endif
        if (0.5*alph*x1norm_sq > x2[0]) {
            l = 2.0/(alph*alph);
            y1norm = solve_reduced_monic_cubic(l*(1 - x2[0]*alph), -l*x1norm);
            x1norm = y1norm/x1norm;
            for (mm = 0; mm < M; mm++) {
                x1[mm] *= x1norm;
            }
            x2[0] = 0.5*alph*y1norm*y1norm;
        }
    } else if (0.0 > x2[0]) {
        x2[0] = 0.0;
    }

#ifdef USE_SHIFT
    for (mm = 0; mm < M+1; mm++) {
        x[i*(M+1) + mm] += shift[i*(M+1) + mm];
    }
#endif
}
#endif

#ifdef EPIGRAPH_PROJ
#include <stdio.h>

inline __device__ TYPE_T compute_Ax(TYPE_T *A, TYPE_T *x) {
    TYPE_T Ax = 0.0;
    Ax = -x[ndim];
    for (int k = 0; k < ndim; k++) {
        Ax += A[k]*x[k];
    }
    return Ax;
}

inline __device__ TYPE_T det_2d(TYPE_T a11, TYPE_T a12, TYPE_T a21, TYPE_T a22) {
    return a11*a22 - a12*a21;
}

inline __device__ TYPE_T det_3d(TYPE_T a11, TYPE_T a12, TYPE_T a13,
                     TYPE_T a21, TYPE_T a22, TYPE_T a23,
                     TYPE_T a31, TYPE_T a32, TYPE_T a33) {
    return a11*det_2d(a22, a23, a32, a33)
         - a21*det_2d(a12, a13, a32, a33)
         + a31*det_2d(a12, a13, a22, a23);
}

inline __device__ void hodge_3d(TYPE_T *a1, TYPE_T *a2, TYPE_T *res) {
    res[0] =  det_2d(a1[1],    -1, a2[1],    -1);
    res[1] = -det_2d(a1[0],    -1, a2[0],    -1);
    res[2] =  det_2d(a1[0], a1[1], a2[0], a2[1]);
}

inline __device__ void hodge_4d(TYPE_T *a1, TYPE_T *a2, TYPE_T *a3, TYPE_T *res) {
    res[0] = -det_3d(a1[1], a1[2],    -1,
                     a2[1], a2[2],    -1,
                     a3[1], a3[2],    -1);
    res[1] =  det_3d(a1[0], a1[2],    -1,
                     a2[0], a2[2],    -1,
                     a3[0], a3[2],    -1);
    res[2] = -det_3d(a1[0], a1[1],    -1,
                     a2[0], a2[1],    -1,
                     a3[0], a3[1],    -1);
    res[3] =  det_3d(a1[0], a1[1], a1[2],
                     a2[0], a2[1], a2[2],
                     a3[0], a3[1], a3[2]);
}

inline __device__ void proj1(TYPE_T *a, TYPE_T *g)
{
    /* Compute the normal projection of g onto the subspace which
     * is orthogonal to (a,-1).
     *
     * This is equivalent to solving
     *
     *      minimize  0.5*<p,p> - <g,p>
     *          s.t.  <(a,-1),p> = 0.
     *
     * The result is stored in g.
     */
    TYPE_T fac, facn;
    uint k;

    // fac : <(a,-1),g> / <(a,-1),(a,-1)>
    fac = compute_Ax(a, g);
    facn = (-1)*(-1);
    for (k = 0; k < ndim; k++) {
        facn += a[k]*a[k];
    }
    fac /= facn;

    // g -= fac*(a,-1)
    g[ndim] -= fac*(-1);
    for (k = 0; k < ndim; k++) {
        g[k] -= fac*a[k];
    }
}

inline __device__ void proj2(TYPE_T *a0, TYPE_T *a1, TYPE_T *g)
{
    /* Compute the normal projection of g onto the subspace which
     * is orthogonal to span{(a0,-1),(a1,-1)}.
     *
     * This is equivalent to solving
     *
     *      minimize  0.5*<p,p> - <g,p>
     *          s.t.  a0[0]*p[0] + a0[1]*p[1] = p[2],
     *                a1[0]*p[0] + a1[1]*p[1] = p[2].
     *
     * The result is stored in g.
     */
    uint k;

#if (ndim == 2)
    // v : cross product of (a0,-1) and (a1,-1)
    TYPE_T v[3];
    hodge_3d(a0, a1, v);

    // fac : <g,v>/<v,v>
    TYPE_T fac = 0;
    TYPE_T facn = 0;
    for (k = 0; k < ndim+1; k++) {
        fac += v[k]*g[k];
        facn += v[k]*v[k];
    }
    fac /= facn;

    // g = fac*v
    for (k = 0; k < ndim+1; k++) {
        g[k] = fac*v[k];
    }
#elif (ndim == 3)
    TYPE_T fac0, fac1;
    TYPE_T a1o[ndim+1];

    // {(a0,-1),a1o} is orthogonal basis of span{(a0,-1),(a1,-1)}
    fac1 = (-1)*(-1) + a1[0]*a0[0] + a1[1]*a0[1] + a1[2]*a0[2];
    fac0 = (-1)*(-1) + a0[0]*a0[0] + a0[1]*a0[1] + a0[2]*a0[2];
    fac1 /= fac0;
    a1o[ndim] = (-1) - fac1*(-1);
    for (k = 0; k < ndim; k++) {
        a1o[k] = a1[k] - fac1*a0[k];
    }

    // g -= <g,(a0,-1)>/<(a0,-1),(a0,-1)> (a0,-1) + <g,a1o>/<a1o,a1o> a1o
    fac0 = compute_Ax(a0, g)/fac0;
    fac1 = a1o[0]*g[0] + a1o[1]*g[1] + a1o[2]*g[2] + a1o[3]*g[3];
    fac1 /= a1o[0]*a1o[0] + a1o[1]*a1o[1] + a1o[2]*a1o[2] + a1o[3]*a1o[3];
    g[ndim] -= fac0*(-1) + fac1*a1o[ndim];
    for (k = 0; k < ndim; k++) {
        g[k] -= fac0*a0[k] + fac1*a1o[k];
    }
#endif
}

inline __device__ void proj3(TYPE_T *a0, TYPE_T *a1, TYPE_T *a2, TYPE_T *g)
{
    /* Compute the normal projection of g onto the subspace which
     * is orthogonal to span{(a0,-1),(a1,-1),(a2,-1)}.
     *
     * This is equivalent to solving
     *
     *      minimize  0.5*<p,p> - <g,p>
     *          s.t.  a0[0]*p[0] + a0[1]*p[1] = p[2],
     *                a1[0]*p[0] + a1[1]*p[1] = p[2],
     *                a2[0]*p[0] + a2[1]*p[1] = p[2].
     *
     * The result is stored in g.
     */
    int k;

    // v : normal of span{(a0,-1),(a1,-1),(a2,-1)}
    TYPE_T v[ndim+1];
    hodge_4d(a0, a1, a2, v);

    // fac : <g,v>/<v,v>
    TYPE_T fac = 0;
    TYPE_T facn = 0;
    for (k = 0; k < ndim+1; k++) {
        fac += v[k]*g[k];
        facn += v[k]*v[k];
    }
    fac /= facn;

    // g = fac*v
    for (k = 0; k < ndim+1; k++) {
        g[k] = fac*v[k];
    }
}

inline __device__ void base_trafo_2d(TYPE_T *a0, TYPE_T *a1, TYPE_T *g)
{
    /* Express g in terms of {(a0,-1), (a1,-1)}.
     *
     * The result is stored in g so that
     *
     *      g[0]*(a0,-1) + g[1]*(a1,-1) = input
     *
     *      g[1] = -input[ndim] - g[0]
     *      g[0]*(a0 - a1) = input[:ndim] + input[ndim]*a1
     */
#if (ndim == 1)
    g[0] += g[ndim]*a1[0];
    g[0] /= (a0[0] - a1[0]);
#else
    uint k;
    TYPE_T matrix00, fac, rhs;

    for (k = 0; k < ndim; k++) {
        g[k] += g[ndim]*a1[k];
    }

    matrix00 = 0;
    rhs = 0;
    for (k = 0; k < ndim; k++) {
        fac = a0[k] - a1[k];
        matrix00 += fac*fac;
        rhs += g[k]*g[k];
    }
    g[0] = rhs/matrix00;
#endif
    g[1] = -g[ndim] - g[0];
}

inline __device__ bool solve_3x3(TYPE_T *A, TYPE_T *b)
{
    /* Solve a 3x3 linear system of equations.
     *
     * If singular, nothing is written and `false` is returned (else `true`).
     *
     * The result is stored in b.
     *
     * This function is an efficient 2x2 solver. Just pass:
     *
     *      A[2] = A[5] = A[6] = A[7] = b[2] = 0.0; A[8] = 1.0;
     *
     * In this case, b still needs to be of length 3!
     *
     */

    TYPE_T fact, a11, a12, a21, a22;
    int row0, row1, row2, tmp;

    // pivoting in first column
    if (FABS(A[0*3 + 0]) >= FABS(A[1*3 + 0])) {
        if (FABS(A[0*3 + 0]) >= FABS(A[2*3 + 0])) {
            row0 = 0; row1 = 1; row2 = 2;
        } else {
            row0 = 2; row1 = 1; row2 = 0;
            fact = b[2]; b[2] = b[0]; b[0] = fact;
        }
    } else if (FABS(A[1*3 + 0]) >= FABS(A[2*3 + 0])) {
        row0 = 1; row1 = 0; row2 = 2;
        fact = b[1]; b[1] = b[0]; b[0] = fact;
    } else {
        row0 = 2; row1 = 1; row2 = 0;
        fact = b[2]; b[2] = b[0]; b[0] = fact;
    }

    if (FABS(A[row0*3 + 0]) < 1e-9) {
        printf("Warning: Bad pivot in solve_3x3, A[0,0]=%g\n", A[row0*3 + 0]);
        return false;
    }

    a11 = A[row1*3 + 1];
    a12 = A[row1*3 + 2];
    a21 = A[row2*3 + 1];
    a22 = A[row2*3 + 2];

    // elimination in first column
    if (A[row1*3 + 0] != 0.0) {
        fact = A[row1*3 + 0]/A[row0*3 + 0];
        a11 -= fact*A[row0*3 + 1];
        a12 -= fact*A[row0*3 + 2];
        b[1] -= fact*b[0];
    }
    if (A[row2*3 + 0] != 0.0) {
        fact = A[row2*3 + 0]/A[row0*3 + 0];
        a21 -= fact*A[row0*3 + 1];
        a22 -= fact*A[row0*3 + 2];
        b[2] -= fact*b[0];
    }

    // pivoting in second column
    if (FABS(a11) < FABS(a21)) {
        tmp = row2; row2 = row1; row1 = tmp;
        fact = a21; a21 = a11; a11 = fact;
        fact = a22; a22 = a12; a12 = fact;
        fact = b[1]; b[1] = b[2]; b[2] = fact;
    }

    if (FABS(a11) < 1e-9) {
        printf("Warning: Bad pivot in solve_3x3, A[1,1]=%g\n", a11);
        return false;
    }

    // elimination in second column
    if (a21 != 0.0) {
        fact = a21/a11;
        a22 -= fact*a12;
        b[2] -= fact*b[1];
    }

    // back substitution
    if (b[2] != 0.0) {
        if (FABS(a22) < 1e-9) {
            printf("Warning: Bad pivot in solve_3x3, A[2,2]=%g\n", a22);
            return false;
        }
        b[2] /= a22;
        b[1] -= a12*b[2];
    }
    b[1] /= a11;
    if (A[row0*3 + 2] != 0.0 && b[2] != 0.0) {
        b[0] -= A[row0*3 + 2]*b[2];
    }
    if (A[row0*3 + 1] != 0.0) {
        b[0] -= A[row0*3 + 1]*b[1];
    }
    b[0] /= A[row0*3 + 0];

    return true;
}

inline __device__ void base_trafo_3d(TYPE_T *a0, TYPE_T *a1, TYPE_T *a2, TYPE_T *g)
{
    /* Express g in terms of {(a0,-1), (a1,-1), (a2,-1)}.
     *
     * The result is stored in g so that
     *
     *      g[0]*(a0,-1) + g[1]*(a1,-1) + g[2]*(a2,-1) = input
     *
     *      g[2] = -input[ndim] - g[0] - g[1]
     *      g[0]*(a0 - a2) + g[1]*(a1 - a2) = input[:ndim] + input[ndim]*a2
     */
    uint k;
    TYPE_T matrix[9];
    TYPE_T g2 = g[2];

#if (ndim == 2)
    for (k = 0; k < 2; k++) {
        g[k] += g2*a2[k];
        matrix[k*3 + 0] = a0[k] - a2[k];
        matrix[k*3 + 1] = a1[k] - a2[k];
    }
    g[2] = 0;
    matrix[2] = 0.0;
    matrix[5] = 0.0;
    matrix[6] = 0.0;
    matrix[7] = 0.0;
    matrix[8] = 1.0;
#else
    g[0] = g[0] + g2*a2[0];
    g[1] = g[1] + g2*a2[1];
    g[2] = g[2] + g2*a2[2];
    matrix[0] = 0; matrix[1] = 0; matrix[4] = 0; matrix[6] = 0; matrix[7] = 0;
    for (k = 0; k < ndim; k++) {
        matrix[0] += (a0[k] - a2[k])*(a0[k] - a2[k]);
        matrix[1] += (a1[k] - a2[k])*(a0[k] - a2[k]);
        matrix[4] += (a1[k] - a2[k])*(a1[k] - a2[k]);
        matrix[6] += g[k]*(a0[k] - a2[k]);
        matrix[7] += g[k]*(a1[k] - a2[k]);
    }

    g[0] = matrix[6];
    g[1] = matrix[7];
    g[2] = 0;
    matrix[2] = 0.0;
    matrix[3] = matrix[1];
    matrix[5] = 0.0;
    matrix[6] = 0.0;
    matrix[7] = 0.0;
    matrix[8] = 1.0;
#endif
    solve_3x3(matrix, g);
    g[2] = -g2 - g[1] - g[0];
}

inline __device__ void base_trafo_4d(TYPE_T *a0, TYPE_T *a1, TYPE_T *a2,
                                     TYPE_T *a3, TYPE_T *g)
{
    /* Express g in terms of {(a0,-1), (a1,-1), (a2,-1), (a3,-1)}.
     *
     * The result is stored in g so that
     *
     *      g[0]*(a0,-1) + g[1]*(a1,-1) + g[2]*(a2,-1) + g[3]*(a3,-1) = input
     *
     *      g[3] = -input[3] - g[0] - g[1] - g[2]
     *      g[0]*(a0 - a3) + g[1]*(a1 - a3) + g[2]*(a2 - a3) = input[:3] + input[3]*a3
     */
    TYPE_T matrix[9];
    for (uint k = 0; k < 3; k++) {
        g[k] += g[3]*a3[k];
        matrix[k*3 + 0] = a0[k] - a3[k];
        matrix[k*3 + 1] = a1[k] - a3[k];
        matrix[k*3 + 2] = a2[k] - a3[k];
    }
    solve_3x3(matrix, g);
    g[3] = -g[3] - g[0] - g[1] - g[2];
}

inline __device__ int array_index_of(int *array, int array_size, int val) {
    for (int i = 0; i < array_size; i++) {
        if (array[i] == val) {
            return i;
        }
    }
    return -1;
}

inline __device__ int array_argmin(TYPE_T *array, int array_size) {
    TYPE_T min = array[0];
    int argmin = 0;
    for (int i = 1; i < array_size; i++) {
        if (array[i] < min) {
            min = array[i];
            argmin = i;
        }
    }
    return argmin;
}

inline __device__ void solve_qp(TYPE_T *x, TYPE_T **A, TYPE_T *b, int N, TYPE_T *sol)
{
    /* This function solves
     *
     *      minimize  0.5*||y - x||**2   s.t.  A y <= b,
     *
     * using an active set method that assumes that at most (ndim+1) constraints
     * are active at the same time.
     *
     * For more details see Algorithm 16.3 in
     *
     *      Nocedal, Wright: Numerical Optimization (2nd Ed.). Springer, 2006.
     *
     *
     *  Args:
     *      x : shape (ndim+1,)
     *      A : shape (N,ndim); the matrix A is actually of shape (N,ndim+1),
     *          but the last column is not stored in memory because it has the
     *          constant value -1
     *      b : shape (N,)
     *      N : number of inequality constraints
     *      sol : shape (ndim+1,), the result is stored in `sol`
     */

    // iteration variables
    int k, l, _iter;

    // dir : search direction
    TYPE_T dir[ndim+1];
    TYPE_T step_min, step, Ax, Ad;

    // active and working set
    int active_set[ndim+1];
    int active_set_size = 0;
    int blocking = -1;

    // lagrange multipliers
    TYPE_T lambda[ndim+1];
    int lambda_argmin;

    // initialize with input
    for (k = 0; k < ndim+1; k++) {
        sol[k] = x[k];
    }

    // determine initial feasible guess by increasing sol[ndim] if necessary
    for (l = 0; l < N; l++) {
        Ax = compute_Ax(A[l], sol);
        if (Ax > b[l]) {
            sol[ndim] += Ax - b[l];
            active_set[0] = l;
            active_set_size = 1;
        }
    }

    // projections are idempotent (feasible inputs are mapped to themselves)
    if (active_set_size == 0) return;

    for (_iter = 0; _iter < term_maxiter; _iter++) {
        blocking = -1;

        // explicitely solve equality constrained helper QPs
        for (k = 0; k < ndim+1; k++) {
            dir[k] = x[k] - sol[k];
        }

        if (active_set_size == 1) {
            proj1(A[active_set[0]], dir);
        } else if (active_set_size == 2) {
            proj2(A[active_set[0]], A[active_set[1]], dir);
        } else if (active_set_size == 3) {
            proj3(A[active_set[0]], A[active_set[1]], A[active_set[2]], dir);
        }

        step = 0.0;
        for (k = 0; k < ndim+1; k++) {
            step += FABS(dir[k]);
        }

        if (step > 0) {
            // determine smallest step size at which a new (blocking)
            // constraint enters the active set
            step_min = 1.0;

            // iterate over constraints not in active set
            for (k = 0; k < N; k++) {
                if (-1 != array_index_of(active_set, active_set_size, k)) {
                    continue;
                }

                Ax = compute_Ax(A[k], sol);
                Ad = compute_Ax(A[k], dir);

                // dir is orthogonal to the active constraints. However, by the
                // following check, dir can't be orthogonal to a blocking
                // constraint. Hence, the blocking constraint is linearly
                // independent from the active constraints.
                if (Ad > term_tolerance) {
                    step = (b[k] - Ax)/Ad;
                    if (step < step_min && step > -term_tolerance) {
                        step_min = step;
                        blocking = k;
                    }
                }
            }

            // advance
            for (k = 0; k < ndim+1; k++) {
                sol[k] += step_min*dir[k];
            }
        }

        if (blocking != -1) {
            // add blocking constraint to active set
            active_set[active_set_size++] = blocking;
        } else if (active_set_size == 1) {
            // no blocking constraint and only one active constraint means
            // we are at the exact orthogonal projection inside of a facet or
            // all blocking constraints were sorted out via Lagrange multipliers
            break;
        }

        if (active_set_size == ndim+1 || blocking == -1) {
            // compute Lagrange multipliers lambda
            for (k = 0; k < ndim+1; k++) {
                lambda[k] = x[k] - sol[k];
            }

            if (active_set_size == 2) {
                base_trafo_2d(A[active_set[0]], A[active_set[1]], lambda);
            } else if (active_set_size == 3) {
                base_trafo_3d(A[active_set[0]], A[active_set[1]],
                              A[active_set[2]], lambda);
            } else if (active_set_size == 4) {
                base_trafo_4d(A[active_set[0]], A[active_set[1]],
                              A[active_set[2]], A[active_set[3]], lambda);
            }

            lambda_argmin = array_argmin(lambda, active_set_size);
            if (lambda[lambda_argmin] >= 0.0) {
                // KKT conditions of full problem are satisfied
                break;
            } else {
                // remove most negative lambda from active set
                active_set[lambda_argmin] = active_set[--active_set_size];
            }
        }
    }

    if (_iter == term_maxiter) {
        printf("Warning: active set method didn't converge within %d "
               "iterations.\n", term_maxiter);
    }

#if 0
    // check feasibility of result
    for (l = 0; l < N; l++) {
        Ax = compute_Ax(A[l], sol);
        if (Ax - b[l] > 1e-3) {
            printf("Warning: solution is not primal feasible: "
                   "diff=%g.\n", Ax - b[l]);
            break;
        }
    }
#endif
}

__global__ void epigraphproj(TYPE_T *x)
{
    /* This function solves, for fixed j and i,
     *
     *      minimize  0.5*||y - x[j,i]||**2
     *          s.t.  A[i,j] y <= b[i,j],
     *
     * using an active set method. The result is stored in x[j,i].
     */

    // global thread index
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    int j = blockIdx.y*blockDim.y + threadIdx.y;

    // stay inside maximum dimensions
    if (j >= nregions || i >= nfuns) return;

    int k, l;
    TYPE_T result[ndim+1];
    TYPE_T *xji = &x[(j*nfuns + i)*(ndim+1)];
    TYPE_T *Aij[nsubpoints];
    TYPE_T bij[nsubpoints];

    // set up constraints
    int Nij = 0;
    for (k = 0; k < nsubpoints; k++) {
        l = J[j*nsubpoints + k];
        if (I[i*npoints + l]) {
            Aij[Nij] = &A_STORE[l*ndim];
            bij[Nij++] = B_STORE[i*npoints + l];
        }
    }

    // solve and write result to input array
    solve_qp(xji, Aij, bij, Nij, result);
    for (k = 0; k < ndim+1; k++) {
        xji[k] = result[k];
    }
}
#endif
