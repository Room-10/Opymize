
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

#ifdef EPIGRAPH_PROJ
#include <stdio.h>

inline __device__ void proj_plane(TYPE_T *a, TYPE_T *g)
{
    /* Compute the normal projection of g onto the subspace which
     * is orthogonal to (a,-1).
     *
     * This is equivalent to solving
     *
     *      minimize  0.5*<p,p> - <g,p>
     *          s.t.  a[0]*p[0] + a[1]*p[1] = p[2].
     *
     * The result is stored in g.
     */

    // fac : <(a,-1),g> / <(a,-1),(a,-1)>
    TYPE_T fac  = a[0]*g[0] + a[1]*g[1] + (-1)*g[2];
           fac /= a[0]*a[0] + a[1]*a[1] + (-1)*(-1);

    // g -= fac*(a,-1)
    g[0] -= fac*a[0];
    g[1] -= fac*a[1];
    g[2] -= fac*(-1);
}

inline __device__ void proj_line(TYPE_T *a0, TYPE_T *a1, TYPE_T *g)
{
    /* Compute the normal projection of g onto the 1-dimensional subspace which
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

    // v : cross product of (a0,-1) and (a1,-1)
    TYPE_T v[3];
    v[0] = a0[1]*(-1)  -  (-1)*a1[1];
    v[1] =  (-1)*a1[0] - a0[0]*(-1) ;
    v[2] = a0[0]*a1[1] - a0[1]*a1[0];

    // fac : <g,v>/<v,v>
    TYPE_T fac  = v[0]*g[0] + v[1]*g[1] + v[2]*g[2];
           fac /= v[0]*v[0] + v[1]*v[1] + v[2]*v[2];

    // g = fac*v
    g[0] = fac*v[0];
    g[1] = fac*v[1];
    g[2] = fac*v[2];
}

inline __device__ void base_trafo_2d(TYPE_T *a0, TYPE_T *a1, TYPE_T *g)
{
    /* Express g in terms of {(a0,-1), (a1,-1)}.
     *
     * The result is stored in g so that
     *
     *      g[0]*(a0,-1) + g[1]*(a1,-1) = input
     */

    TYPE_T diff0 = a0[0] - a1[0];
    TYPE_T diff1 = a0[1] - a1[1];

    if (FABS(diff1) > FABS(diff0)) {
        g[0] = (g[1] + g[2]*a1[1])/diff1;
    } else {
        g[0] = (g[0] + g[2]*a1[0])/diff0;
    }

    // make use of -mu[0]-mu[1] = g[2]
    g[1] = -g[2] - g[0];
}

inline __device__ bool solve_2x2(TYPE_T *A, TYPE_T *b)
{
    /* Solve a 2x2 linear system of equations.
     *
     * If singular, nothing is written and `false` is returned (else `true`).
     *
     * The result is stored in b.
     */

    TYPE_T detA = A[0]*A[3] - A[1]*A[2];
    TYPE_T res0, res1;
    int row0 = 0;
    int row1 = 1;

    if (FABS(detA) < 1e-9) {
        printf("Warning: Singular matrix in solve_2x2, det(A)=%g\n", detA);
        return false;
    } else {
        if(FABS(A[row0*2 + 0]) < FABS(A[row1*2 + 0])) {
            // swap rows for numerical stability
            row0 = 1; row1 = 0;
        }
        res1 = A[row1*2 + 0]/A[row0*2 + 0];
        res0 = A[row1*2 + 1] - A[row0*2 + 1]*res1;
        res1 = (b[row1] - b[row0]*res1)/res0;
        res0 = (b[row0] - A[row0*2 + 1]*res1)/A[row0*2 + 0];
        b[0] = res0;
        b[1] = res1;
        return true;
    }
}

inline __device__ void base_trafo_3d(TYPE_T *a0, TYPE_T *a1, TYPE_T *a2, TYPE_T *g)
{
    /* Express g in terms of {(a0,-1), (a1,-1), (a2,-1)}.
     *
     * The result is stored in g so that
     *
     *      g[0]*(a0,-1) + g[1]*(a1,-1) + g[2]*(a2,-1) = input
     */

    TYPE_T matrix[4];
    matrix[0] = a0[0] - a2[0];
    matrix[1] = a1[0] - a2[0];
    matrix[2] = a0[1] - a2[1];
    matrix[3] = a1[1] - a2[1];
    g[0] = g[0] + g[2]*a2[0];
    g[1] = g[1] + g[2]*a2[1];
    solve_2x2(matrix, g);

    // make use of -mu[0]-mu[1]-mu[2] = g[2]
    g[2] = -g[2] - g[1] - g[0];
}

inline __device__ int array_index_of(int *array, int array_size, int val) {
    for (int i = 0; i < array_size; i++) {
        if (array[i] == val) {
            return i;
        }
    }
    return -1;
}

inline __device__ int array_argmin(TYPE_T *array, TYPE_T array_size) {
    TYPE_T min = array[0];
    TYPE_T argmin = 0;
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
     * using an active set method that assumes that at most three constraints
     * are active at the same time.
     *
     * For more details see Algorithm 16.3 in
     *
     *      Nocedal, Wright: Numerical Optimization (2nd Ed.). Springer, 2006.
     *
     *
     *  Args:
     *      x : shape (3,)
     *      A : shape (N,2); the matrix A is actually of shape (N,3), but the
     *          last column is not stored in memory because it has the constant
     *          value -1
     *      b : shape (N,)
     *      N : number of inequality constraints
     *      sol : shape (3,), the result is stored in `sol`
     */

    // iteration variables
    int k, l, _iter;

    // dir : search direction
    TYPE_T dir[3];
    TYPE_T step_min, step, Ax, Ad;

    // active and working set
    int active_set[3];
    int active_set_size = 0;
    int blocking = -1;

    // lagrange multipliers
    TYPE_T lambda[3];
    int lambda_argmin;

    // initialize with input
    for (k = 0; k < 3; k++) {
        sol[k] = x[k];
    }

    // determine initial feasible guess by increasing sol[2] if necessary
    for (l = 0; l < N; l++) {
        Ax = A[l][0]*sol[0] + A[l][1]*sol[1] - sol[2];
        if (Ax > b[l]) {
            sol[2] += Ax - b[l];
            active_set[0] = l;
            active_set_size = 1;
        }
    }

    // projections are idempotent (feasible inputs are mapped to themselves)
    if (active_set_size == 0) return;

    for (_iter = 0; _iter < term_maxiter; _iter++) {
        blocking = -1;

        // explicitely solve equality constrained helper QPs
        for (k = 0; k < 3; k++) {
            dir[k] = x[k] - sol[k];
        }

        if (active_set_size == 1) {
            proj_plane(A[active_set[0]], dir);
        } else if (active_set_size == 2) {
            proj_line(A[active_set[0]], A[active_set[1]], dir);
        }

        if (FABS(dir[0]) + FABS(dir[1]) + FABS(dir[2]) > 0) {
            // determine smallest step size at which a new (blocking)
            // constraint enters the active set
            step_min = 1.0;

            // iterate over constraints not in active set
            for (k = 0; k < N; k++) {
                if (-1 != array_index_of(active_set, active_set_size, k)) {
                    continue;
                }

                Ax = A[k][0]*sol[0]   + A[k][1]*sol[1]   - sol[2];
                Ad = A[k][0]*dir[0] + A[k][1]*dir[1] - dir[2];

                // dir is orthogonal to a0 and a1. However, by the following
                // check, dir can't be orthogonal to a blocking constraint,
                // hence (a0,a1,a2) is always linearly independent.
                if (Ad > term_tolerance) {
                    step = (b[k] - Ax)/Ad;
                    if (step < step_min && step > -term_tolerance) {
                        step_min = step;
                        blocking = k;
                    }
                }
            }

            // advance
            for (k = 0; k < 3; k++) {
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

        if (active_set_size == 3 || blocking == -1) {
            // compute Lagrange multipliers lambda
            lambda[0] = x[0] - sol[0];
            lambda[1] = x[1] - sol[1];
            lambda[2] = x[2] - sol[2];

            if (active_set_size == 2) {
                // No blocking constraint: sol is exact orthogonal projection of
                // x onto orth{a0,a1}. Hence, x-sol is in span{a0,a1}.
                base_trafo_2d(A[active_set[0]], A[active_set[1]], lambda);
            } else if (active_set_size == 3) {
                // dir != 0 and a blocking constraint a2. In this case,
                // (a0,a1,a2) is linearly independent (see comment above).
                base_trafo_3d(A[active_set[0]], A[active_set[1]],
                              A[active_set[2]], lambda);
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
        Ax = A[l][0]*sol[0] + A[l][1]*sol[1] - sol[2];
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
    int j = blockIdx.x*blockDim.x + threadIdx.x;
    int i = blockIdx.y*blockDim.y + threadIdx.y;

    // stay inside maximum dimensions
    if (j >= nregions || i >= nfuns) return;

    int k, l;
    TYPE_T result[3];
    TYPE_T *xji = &x[(j*nfuns + i)*3];
    TYPE_T *Aij[nsubpoints];
    TYPE_T bij[nsubpoints];

    // set up constraints
    int Nij = 0;
    for (k = 0; k < nsubpoints; k++) {
        l = J[j*nsubpoints + k];
        if (I[i*npoints + l]) {
            Aij[Nij] = &A_STORE[l*2];
            bij[Nij++] = B_STORE[i*npoints + l];
        }
    }

    // solve and write result to input array
    solve_qp(xji, Aij, bij, Nij, result);
    for (k = 0; k < 3; k++) {
        xji[k] = result[k];
    }
}
#endif
