
#ifdef MATRIX_MULT
__global__ void matrixmult(double *x, double *y)
{
    /* y_ji = \sum_k A_jk * x_ki */

    // global thread index
    int j = blockIdx.x*blockDim.x + threadIdx.x;
    int i = blockIdx.y*blockDim.y + threadIdx.y;

    // stay inside maximum dimensions
    if(j >= J || i >= N) return;

    // iteration variables and misc.
    int kk, idx;
    double newval;

    idx = j*N + i;
    newval = y[idx];
    for(kk = 0; kk < K; kk++) {
#if trans == 'n'
        newval += A[j*K + kk]*x[kk*N + i];
#else
        newval += A[kk*J + j]*x[kk*N + i];
#endif
    }
    y[idx] = newval;
}
#endif

#ifdef MATRIX_MULT_R
__global__ void matrixmultr(double *x, double *y)
{
    /* y_ij = \sum_k x_ik * A_kj */

    // global thread index
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    int j = blockIdx.y*blockDim.y + threadIdx.y;

    // stay inside maximum dimensions
    if(i >= N || j >= J) return;

    // iteration variables and misc.
    int kk, idx;
    double newval;

    idx = i*J + j;
    newval = y[idx];
    for(kk = 0; kk < K; kk++) {
#if trans == 'n'
        newval += x[i*K + kk]*A[kk*J + j];
#else
        newval += x[i*K + kk]*A[j*K + kk];
#endif
    }
    y[idx] = newval;
}
#endif

#ifdef MATRIX_MULT_R_BATCHED
__global__ void matrixmultrbatched(double *x, double *y)
{
    /* y_jlk = \sum_m x_jlm * A_jmk */

    // global thread index
    int j = blockIdx.x*blockDim.x + threadIdx.x;
    int l = blockIdx.y*blockDim.y + threadIdx.y;
    int k = blockIdx.z*blockDim.z + threadIdx.z;

    // stay inside maximum dimensions
    if(j >= J || l >= L || k >= K) return;

    // iteration variables and misc.
    int mm, idx;
    double newval;

    idx = j*(L*K) + l*K + k;
    newval = y[idx];
    for(mm = 0; mm < M; mm++) {
#if trans == 'n'
        newval += x[j*(L*M) + l*M + mm]*A[j*(M*K) + mm*K + k];
#else
        newval += x[j*(L*M) + l*M + mm]*A[j*(M*K) + k*M + mm];
#endif
    }
    y[idx] = newval;
}
#endif

#ifdef TANGLED_MATRIX_MULT_R
__global__ void tangledmatrixmultr(double *x, double *y)
{
    /* y_mik = \sum_jl x_jil * A_jlmk */

    // global thread index
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    int mk = blockIdx.y*blockDim.y + threadIdx.y;

    // stay inside maximum dimensions
    if(i >= N || mk >= MK) return;

    // disentangle index
    int m = mk / K;
    int k = mk % K;

    // iteration variables and misc.
    int jj, ll, idx;
    double newval;

    idx = m*(N*K) + i*K + k;
    newval = y[idx];
    for(jj = 0; jj < J; jj++) {
        for(ll = 0; ll < L; ll++) {
#if trans == 'n'
            newval += x[jj*(N*L) + i*L + ll]*A[jj*(L*MK) + ll*MK + mk];
#else
            newval += x[jj*(N*L) + i*L + ll]*A[mk*(J*L) + jj*L + ll];
#endif
        }
    }
    y[idx] = newval;
}
#endif
