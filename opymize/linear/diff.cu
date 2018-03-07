
__device__ int* i2coords(int i) {
    int aa;
    int coords[D];
    for (aa = D - 1; aa >= 0; aa--) {
        coords[D - 1 - aa] = i / skips[aa];
        i = i % skips[aa];
    }
    return coords;
}

__device__ int is_br_boundary(int i) {
    int aa;
    int* coords = i2coords(i);
    for (aa = 0; aa < D; aa++) {
        if (imagedims[aa]-1 == coords[aa]) {
            return true;
        }
    }
    return false;
}

__device__ int avgskip_allowed(int t, int *coords, int d) {
    int aa, dk;
    for (aa = D - 1; aa >= 0; aa--) {
        dk = d / skips[aa];
        d = d % skips[aa];
        if (aa == t) continue;
        dk = coords[D - 1 - aa] - dk;
        if (dk < 0 || dk == imagedims[D - 1 - aa]-1) {
            return false;
        }
    }
    return true;
}

__global__ void gradient(double *x, double *y)
{
    // y += D x (D is the gradient on a staggered grid with Neumann boundary)

    // global thread index
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    int k = blockIdx.y*blockDim.y + threadIdx.y;
    int t = blockIdx.z*blockDim.z + threadIdx.z;

    // stay inside maximum dimensions
    if (i >= N || k >= C || t >= D) return;

    // iteration variable and misc.
    int aa, base;
    double newval, fac;

    newval = 0.0;
    fac = weights[k]/(double)navgskips;

    // skip points on "bottom right" boundary
    if (!is_br_boundary(i)) {
        for (aa = 0; aa < navgskips; aa++) {
            base = i + avgskips[t*navgskips + aa];
            newval += x[(base + skips[t])*C + k] - x[base*C + k];
        }
    }

    y[i*dc_skip + t*C + k] += fac*newval;
}

__global__ void divergence(double *x, double *y)
{
    // y += D' x (D' = -div with Dirichlet boundary)

    // global thread index
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    int k = blockIdx.y*blockDim.y + threadIdx.y;

    // stay inside maximum dimensions
    if (i >= N || k >= C)
       return;

    // iteration variable and misc.
    int tt, aa, base, idx;
    int* coords = i2coords(i);
    double newval, fac;

    newval = 0.0;
    fac = weights[k]/(double)navgskips;

    for (tt = 0; tt < D; tt++) {
        idx = tt*C + k;
        for (aa = 0; aa < navgskips; aa++) {
            if(avgskip_allowed(tt, coords, avgskips[tt*navgskips + aa])) {
                base = i - avgskips[tt*navgskips + aa];
                if (coords[D - 1 - tt] < imagedims[D - 1 - tt]-1) {
                    newval -= x[idx + base*dc_skip];
                }
                if (coords[D - 1 - tt] > 0) {
                    newval += x[idx + (base - skips[tt])*dc_skip];
                }
            }
        }
    }

    y[i*C + k] += fac*newval;
}
