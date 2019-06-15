
inline __device__ void i2coords(int i, int *coords) {
    int tt;
    for (tt = 0; tt < D; tt++) {
        coords[tt] = i / skips[tt];
        i = i % skips[tt];
    }
}

#ifdef GRAD_DIV
inline __device__ int is_br_boundary(int i) {
    int aa;
    int coords[D];
    i2coords(i, coords);
    for (aa = 0; aa < D; aa++) {
        if (imagedims[aa]-1 == coords[aa]) {
            return true;
        }
    }
    return false;
}

inline __device__ int avgskip_allowed(int t, int *coords, int d) {
    int aa, dk;
    for (aa = 0; aa < D; aa++) {
        dk = d / skips[aa];
        d = d % skips[aa];
        if (aa == t) continue;
        dk = coords[aa] - dk;
        if (dk < 0 || dk == imagedims[aa]-1) {
            return false;
        }
    }
    return true;
}

__global__ void gradient(TYPE_T *x, TYPE_T *y)
{
    // y += D x (D is the gradient on a staggered grid with Neumann boundary)

    // global thread index
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    int k = blockIdx.y*blockDim.y + threadIdx.y;
    int t = blockIdx.z*blockDim.z + threadIdx.z;

    // stay inside maximum dimensions
    if (i >= N || k >= C || t >= D) return;

    // iteration variable and misc.
    TYPE_T newval, fac;
    newval = 0.0;

#if defined(SCHEME_CENTERED)
    int aa, base;
    fac = weights[k]/(imageh[t]*(TYPE_T)navgskips);

    // skip points on "bottom or right" boundary
    if (!is_br_boundary(i)) {
        for (aa = 0; aa < navgskips; aa++) {
            base = i + avgskips[t*navgskips + aa];
            newval += x[(base + skips[t])*C + k] - x[base*C + k];
        }
    }
#elif defined(SCHEME_FORWARD)
    int coords[D];
    i2coords(i, coords);

    fac = weights[k]/imageh[t];
    if (coords[t] < imagedims[t]-1) {
        newval += x[(i + skips[t])*C + k] - x[i*C + k];
    }
#endif

    y[i*dc_skip + t*C + k] += fac*newval;
}

__global__ void divergence(TYPE_T *x, TYPE_T *y)
{
    // y += D' x (D' = -div with Dirichlet boundary)

    // global thread index
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    int k = blockIdx.y*blockDim.y + threadIdx.y;

    // stay inside maximum dimensions
    if (i >= N || k >= C)
       return;

    // iteration variable and misc.
    int tt;
    int coords[D];
    i2coords(i, coords);
    TYPE_T newval, fac;
    newval = 0.0;

#if defined(SCHEME_CENTERED)
    int aa, base, idx;
    fac = weights[k]/(TYPE_T)navgskips;

    for (tt = 0; tt < D; tt++) {
        idx = tt*C + k;
        for (aa = 0; aa < navgskips; aa++) {
            if(avgskip_allowed(tt, coords, avgskips[tt*navgskips + aa])) {
                base = i - avgskips[tt*navgskips + aa];
                if (coords[tt] < imagedims[tt]-1) {
                    newval -= x[idx + base*dc_skip]/imageh[tt];
                }
                if (coords[tt] > 0) {
                    newval += x[idx + (base - skips[tt])*dc_skip]/imageh[tt];
                }
            }
        }
    }
#elif defined(SCHEME_FORWARD)
    fac = weights[k];
    for (tt = 0; tt < D; tt++) {
        if (coords[tt] < imagedims[tt]-1) {
            newval -= x[i*dc_skip + tt*C + k]/imageh[tt];
        }
        if (coords[tt] > 0) {
            newval += x[(i - skips[tt])*dc_skip + tt*C + k]/imageh[tt];
        }
    }
#endif

    y[i*C + k] += fac*newval;
}
#endif

#ifdef LAPLACIAN
__global__ void laplacian(TYPE_T *x, TYPE_T *y)
{
    // y += \Delta x (\Delta is the Laplacian with various boundary conditions)

    // global thread index
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    int k = blockIdx.y*blockDim.y + threadIdx.y;

    // stay inside maximum dimensions
    if (i >= N || k >= C) return;

    // iteration variable and misc.
    int tt;
    int coords[D];
    i2coords(i, coords);
    TYPE_T newval, xik, invh2;

    xik = x[i*C + k];
    newval = 0;

#if boundary_conditions == 'C'
#if ADJOINT
    // skip corners
    int boundary_dim = -1;
    for (tt = 0; tt < D; tt++) {
        if (coords[tt] == 0 || coords[tt] == imagedims[tt]-1) {
            if (boundary_dim >= 0) {
                return;
            } else {
                boundary_dim = tt;
            }
        }
    }

    if (boundary_dim >= 0) {
        tt = boundary_dim;
        invh2 = 1.0/(imageh[tt]*imageh[tt]);
        if (coords[tt] == 0) {
            newval += invh2*x[(i + skips[tt])*C + k];
        } else if (coords[tt] == imagedims[tt]-1) {
            newval += invh2*x[(i - skips[tt])*C + k];
        }
    } else {
        for (tt = 0; tt < D; tt++) {
            invh2 = 1.0/(imageh[tt]*imageh[tt]);
            newval -= 2*invh2*xik;

            if (coords[tt]-1 > 0) {
                newval += invh2*x[(i - skips[tt])*C + k];
            }

            if (coords[tt]+1 < imagedims[tt]-1) {
                newval += invh2*x[(i + skips[tt])*C + k];
            }
        }
    }
#else // !ADJOINT
    // skip all boundary points entirely
    for (tt = 0; tt < D; tt++) {
        if (coords[tt] == 0 || coords[tt] == imagedims[tt]-1) {
            return;
        }
    }

    for (tt = 0; tt < D; tt++) {
        invh2 = 1.0/(imageh[tt]*imageh[tt]);
        newval += invh2*(x[(i - skips[tt])*C + k] - xik)
               +  invh2*(x[(i + skips[tt])*C + k] - xik);
    }
#endif
#elif boundary_conditions == 'S'
#if ADJOINT
    for (tt = 0; tt < D; tt++) {
        invh2 = 1.0/(imageh[tt]*imageh[tt]);

        if (coords[tt] > 0 && coords[tt] < imagedims[tt]-1) {
            newval -= invh2*2*xik;
        }

        if (coords[tt] > 1) {
            newval += invh2*x[(i - skips[tt])*C + k];
        }

        if (coords[tt] < imagedims[tt]-2) {
            newval += invh2*x[(i + skips[tt])*C + k];
        }
    }
#else // !ADJOINT
    for (tt = 0; tt < D; tt++) {
        invh2 = 1.0/(imageh[tt]*imageh[tt]);

        if (coords[tt] > 0 && coords[tt] < imagedims[tt]-1) {
            newval += invh2*(x[(i - skips[tt])*C + k] - 2*xik
                             + x[(i + skips[tt])*C + k]);
        }
    }
#endif
#else // boundary_conditions == 'N'
    for (tt = 0; tt < D; tt++) {
        invh2 = 1.0/(imageh[tt]*imageh[tt]);

        if (coords[tt] > 0) {
            newval += invh2*(x[(i - skips[tt])*C + k] - xik);
        }

        if (coords[tt] < imagedims[tt]-1) {
            newval += invh2*(x[(i + skips[tt])*C + k] - xik);
        }
    }
#endif

    y[i*C + k] += newval;
}
#endif
