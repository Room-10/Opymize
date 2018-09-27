
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
__global__ void epigraphproj(TYPE_T *X_STORE)
{
    /* This function solves, for fixed i and j,
     *
     *      minimize  0.5*||y - x[i,j]||**2  s.t.  A[i,j] y >= b[i,j]
     *
     * using an active set method.
     *
     * The result is stored in &x[i,j].
     */

    // global thread index
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    int j = blockIdx.y*blockDim.y + threadIdx.y;

    // stay inside maximum dimensions
    if (i >= nfuns || j >= nregions) return;

    int idx = i*nregions + j;

    TYPE_T *A = &A_STORE[idx];
    TYPE_T *b = &B_STORE[idx];
    TYPE_T *x = &X_STORE[idx];

    TYPE_T[3] y;

    // SOLVER MAGIC COMES HERE

    for (int k = 0; k < 3; k++) {
        x[idx + k] = y[k];
    }
}
#endif
