#include <iostream>
#include <cstdlib>
#include <vector>

#include <curand_kernel.h>

#include "cnpy.h"
#include "kernel.cuh"




int main() {
    const int N = 1000; // Number of particles
    const int Nt = 50; // Number of time stpes
    const float dt = 0.25; // Time step

    int bytes = N*sizeof(float); // Size of float array

    int blockSize = 64; // Threads per block
    int gridSize = (N + blockSize - 1)/blockSize ; // Number of block
    
    // RNG initialization
    curandState *states;
    cudaMalloc(&states, N*sizeof(curandState));
    init_rng<<<gridSize, blockSize>>>(states, 10, N);
    cudaDeviceSynchronize();

    // x Initialization
    float *x, *x_t;
    cudaMalloc(&x_t, bytes); // Current x_i(t)
    cudaMalloc(&x, Nt*bytes); // Total trajectories of x_i
    init_state<<<gridSize, blockSize>>>(x_t, states, 0.0, 1.0, N);
    cudaDeviceSynchronize();

    for (int t = 0; t < Nt; ++t) {
        cudaMemcpy(x + t*N, x_t, bytes, cudaMemcpyDeviceToDevice);
        EulerMaruyama<<<gridSize, blockSize>>>(x_t, states, dt, N);
    }

    cudaDeviceSynchronize();

    std::vector<float> h_x(Nt*N);
    std::vector<size_t> shape = {Nt, N};

    cudaMemcpy(h_x.data(), x, Nt*bytes, cudaMemcpyDeviceToHost);

    cnpy::npy_save("Langevin.npy", h_x.data(), shape, "w");

    cudaFree(x_t);
    cudaFree(x);
    cudaFree(states);

    return 0;
}
