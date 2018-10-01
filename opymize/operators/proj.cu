
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
        g[1] = (g[1] + g[2]*a1[1])/diff1;
    } else {
        g[1] = (g[0] + g[2]*a1[0])/diff0;
    }

    // make use of -mu[0]-mu[1] = g[2]
    g[0] = - g[2] - g[1];
}

inline __device__ void swap_vars(TYPE_T *a, TYPE_T *b)
{
  TYPE_T c = a[0]; a[0] = b[0]; b[0] = c;
}

inline __device__ void solve_2x2(TYPE_T *A, TYPE_T *b)
{
    /* Solve a 2x2 linear system of equations.
     *
     * If singular, the projection of b onto span{a0} is returned, i.e.:
     *
     *      b[0] = <b,a0>/<a0,a0>,   b[1] = 0.
     *
     * The result is stored in b.
     */

    TYPE_T detA = A[0]*A[3] - A[1]*A[2];
    TYPE_T alpha, beta, gamma;

    if (FABS(detA) < 1e-9) {
        printf("Warning: Singular matrix in solve_2x2, det(A)=%g\n", detA);
        b[0] = (b[0]*A[0] + b[1]*A[2])/(A[0]*A[0] + A[2]*A[2]);
        b[1] = 0;
    } else {
        if(FABS(A[0]) < FABS(A[2])) {
            swap_vars(&A[2], &A[0]);
            swap_vars(&A[3], &A[1]);
            swap_vars(&b[0], &b[1]);
        }
        alpha = A[2]/A[0];
        beta = A[3] - A[1]*alpha;
        gamma = b[1] - b[0]*alpha;
        b[1] = gamma/beta;
        b[0] = (b[0] - A[1]*b[1])/A[0];
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

__global__ void epigraphproj(TYPE_T *x)
{
    /* This function solves, for fixed j and i,
     *
     *      minimize  0.5*||y - x[j,i]||**2
     *          s.t.  A[i,j] y <= b[i,j],
     *
     * using an active set method that assumes that at most three constraints
     * are active at the same time (which can be satisfied in the case where the
     * A[i,j] come from polyhedral epigraphs of convex functions).
     *
     * For more details see Algorithm 16.3 in
     *
     *      Nocedal, Wright: Numerical Optimization (2nd Ed.). Springer, 2006.
     *
     * The matrices A are of shape (N,3), but the last column is not stored in
     * memory because it has the constant value -1.
     *
     * The result is stored in x[j,i].
     */

    // global thread index
    int j = blockIdx.x*blockDim.x + threadIdx.x;
    int i = blockIdx.y*blockDim.y + threadIdx.y;

    // stay inside maximum dimensions
    if (j >= nregions || i >= nfuns) return;

    // iteration variables
    int k, l, _iter;

    // N : number of inequality constraints
    // A : shape (N,2)
    // b : shape (N,)
    // xji : shape (3,)
    int N = counts[i*nregions + j];
    TYPE_T *A = &A_STORE[2*indices[i*nregions + j]];
    TYPE_T *b = &B_STORE[indices[i*nregions + j]];
    TYPE_T *xji = &x[(j*nfuns + i)*3];

    // y : iteration variable
    // dir : search direction
    TYPE_T y[3];
    TYPE_T dir[3];
    TYPE_T min_step, step, Ax, Ad;

    // active and working set
    int active_set[3];
    int active_set_size = 0;
    int blocking = -1;
    bool in_set;

    // lagrange multipliers
    TYPE_T lambda[3];
    TYPE_T lambda_best;
    int lambda_best_idx;

    // initialize with input
    for (k = 0; k < 3; k++) {
        y[k] = xji[k];
    }

    // determine initial feasible guess by increasing y[2] if necessary
    for (l = 0; l < N; l++) {
        Ax = A[l*2 + 0]*y[0] + A[l*2 + 1]*y[1];
        if (Ax - y[2] > b[l]) {
            y[2] = Ax - b[l];
            active_set[0] = l;
            active_set_size = 1;
        }
    }

    // projections are idempotent (feasible inputs are mapped to themselves)
    if (active_set_size == 0) return;

    for (_iter = 0; _iter < term_maxiter; _iter++) {
        blocking = -1;

        if (active_set_size > 2) {
            // Whenever the active set reaches a size of three, it is reduced
            // at the end of the iteration, according to the current Lagrange
            // multipliers.
            printf("Warning: active_set_size=%d at new iter.\n", active_set_size);
        } else {
            // explicitely solve equality constrained helper QPs
            for (k = 0; k < 3; k++) {
                dir[k] = xji[k] - y[k];
            }
            if (active_set_size == 1) {
                proj_plane(&A[active_set[0]*2], dir);
            } else if (active_set_size == 2) {
                proj_line(&A[active_set[0]*2], &A[active_set[1]*2], dir);
            }

            if (FABS(dir[0]) + FABS(dir[1]) + FABS(dir[2]) > 0) {
                // determine smallest step size at which a new (blocking)
                // constraint enters the active set
                min_step = 1.0;

                // iterate over constraints not in active set
                for (k = 0; k < N; k++) {
                    in_set = false;
                    for (l = 0; l < active_set_size; l++) {
                        if (active_set[l] == k) { in_set = true; break; }
                    }
                    if (in_set) continue;

                    Ax = A[k*2 + 0]*y[0]   + A[k*2 + 1]*y[1]   - y[2];
                    Ad = A[k*2 + 0]*dir[0] + A[k*2 + 1]*dir[1] - dir[2];

                    if (Ad > term_tolerance) {
                        step = (b[k] - Ax)/Ad;
                        if (step < min_step && step > -term_tolerance) {
                            min_step = step;
                            blocking = k;
                        }
                    }
                }

                // advance
                for (k = 0; k < 3; k++) {
                    y[k] += min_step * dir[k];
                }
            }

            if (blocking != -1) {
                // add blocking constraint to active set
                active_set[active_set_size++] = blocking;
            } else if (active_set_size == 1) {
                // moved freely without blocking constraint and at least one
                // constraint is active at solution -> convergence
                break;
            }
        }

        if (active_set_size == 3 || blocking == -1) {
            // compute Lagrange multipliers lambda
            lambda[0] = xji[0] - y[0];
            lambda[1] = xji[1] - y[1];
            lambda[2] = xji[2] - y[2];
            if (active_set_size == 2) {
                // this is solvable for any RHS due to the properties of A
                base_trafo_2d(&A[active_set[0]*2], &A[active_set[1]*2], lambda);
            } else if (active_set_size == 3) {
                // only way to get here is dir != 0 and a blocking constraint
                // always solvable?
                base_trafo_3d(&A[active_set[0]*2], &A[active_set[1]*2],
                              &A[active_set[2]*2], lambda);
            } else {
                printf("Warning: unusual active set size %d.\n", active_set_size);
            }

            // Check positivity of lambda
            lambda_best = 0.0;
            lambda_best_idx = -1;
            for (k = 0; k < active_set_size; k++) {
                if (lambda[k] < lambda_best) {
                    lambda_best = lambda[k];
                    lambda_best_idx = k;
                }
            }

            if (lambda_best_idx == -1) {
                // converged (all Lagrange multipliers in active set positive)
                break;
            } else {
                // remove most negative lambda from active set
                active_set[lambda_best_idx] = active_set[--active_set_size];
            }
        }
    }

#if 0
    if (_iter == term_maxiter) {
        printf("Warning: active set method didn't converge within %d iterations "
               "at (%d,%d).\n", term_maxiter, i, j);
    }

    // check feasibility of result
    for (l = 0; l < N; l++) {
        Ax = A[l*2 + 0]*y[0] + A[l*2 + 1]*y[1] - y[2];
        if (Ax - b[l] > 1e-3) {
            printf("Warning: solution is not primal feasible at (%d,%d): "
                   "diff=%g.\n", i, j, Ax - b[l]);
            break;
        }
    }
#endif

    // write result to input array
    for (k = 0; k < 3; k++) {
        xji[k] = y[k];
    }
}
#endif
