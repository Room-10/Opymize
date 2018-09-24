
__global__ void indexedmultadj(TYPE_T *x, TYPE_T *y)
{
    /* y^i -= sum_j P^j' B^j' x^ij */

    // global thread index
    int i = blockIdx.x*blockDim.x + threadIdx.x;

    // stay inside maximum dimensions
    if (i >= N) return;

    // iteration variables and misc.
    int jj, ll, mm, idx;
    TYPE_T newval;

    for (jj = 0; jj < J; jj++) {
        for (ll = 0; ll < L; ll++) {
            idx = i*K + P[jj*L + ll];
            newval = y[idx];
            for (mm = 0; mm < M; mm++) {
                newval -= B[jj*(L*M) + mm*L + ll]*x[jj*(N*M) + i*M + mm];
            }
            y[idx] = newval;
        }
    }
}

__global__ void indexedmult(TYPE_T *x, TYPE_T *y)
{
    /* y^ij -= B^j P^j x^i */

    // global thread index
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    int j = blockIdx.y*blockDim.y + threadIdx.y;
    int m = blockIdx.z*blockDim.z + threadIdx.z;

    // stay inside maximum dimensions
    if (i >= N || j >= J || m >= M) return;

    // iteration variables and misc.
    int idx = j*(N*M) + i*M + m;
    TYPE_T newval = y[idx];

    for (int ll = 0; ll < L; ll++) {
        newval -= B[j*(L*M) + m*L + ll]*x[i*K + P[j*L + ll]];
    }

    y[idx] = newval;
}