#include "kernel.cuh"
#include <curand_kernel.h>
#include <cmath>


/*
__device__ float drift(float x) {
    float F = 0.0;

    return F;
}


__device__ float diffusion(float x) {
    float D = 1.0;

    return sqrtf(2*D);
}
*/

__global__ void init_rng(curandState *states, unsigned long seed, int N){
    int idx = threadIdx.x + blockDim.x*blockIdx.x;

    if (idx >= N) return;

    curand_init(seed, idx, 0, &states[idx]);
}


__global__ void init_state(float *x_t, curandState *states, float m, float s, int N){
    int idx = threadIdx.x + blockDim.x*blockIdx.x;

    if (idx >= N) return;

    x_t[idx] = m + s*curand_normal(&states[idx]);
}


__global__ void EulerMaruyama(float *x_t, curandState *states, const float dt, const int N){
    int idx = threadIdx.x + blockDim.x*blockIdx.x;

    float mu = 0.0; // Drift
    float D = 1.0; // Diffusion coefficient

    if (idx >= N) return;

    float dW = curand_normal(&states[idx])*sqrtf(dt);

    x_t[idx] += mu*dt + sqrtf(2*D)*dW;

}
