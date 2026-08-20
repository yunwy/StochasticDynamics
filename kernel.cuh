#pragma once
#include <curand_kernel.h>


//__device__ float drift(float x);
//__device__ float diffusion(float x);

__global__ void init_rng(curandState *states, unsigned long seed, int N);
__global__ void init_state(float *x_t, curandState *states, float m, float s, int N);
__global__ void EulerMaruyama(float *x_t, curandState *states, const float dt, const int N);
