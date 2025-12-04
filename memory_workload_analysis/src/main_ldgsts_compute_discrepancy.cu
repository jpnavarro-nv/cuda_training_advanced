#include <stdio.h>
#include <stdint.h>
#include "parfileReader.h"
#include <cuda_pipeline_primitives.h>
#include "helpers.h"
#include "fd_coeffs.h"

template <typename T, int radius, int tile_dim0>
__global__ void deriv0_ldg(T const *__restrict__ left_halo,
                           T const *__restrict__ right_halo,
                           T const *__restrict__ center,
                           T *__restrict__ output)
{
    // Declare shared memory array, size is known at compile time
    __shared__ T smem[2 * radius + tile_dim0];

    // Get thread index
    const int tx = threadIdx.x;
    if (tx >= tile_dim0)
        return;

    // Fill temporary shared memory input array
    if (tx < radius)
        smem[tx] = left_halo[tx];

    if (tx >= tile_dim0 - radius)
        smem[2 * radius + tx] = right_halo[radius - tile_dim0 + tx];

    smem[tx + radius] = center[tx];

    // Wait until all threads within block are done writing to shared memory
    __syncthreads();

    // if (tx == 0)
    //     for (int i = 0; i < tile_dim0 + 2 * radius; i++)
    //         printf("i: %d, smem: %f\n", i, smem[i]);

    // Compute derivative and write output to global memory
    T out_tmp = d2coef<radius>(0) * smem[tx + radius];
#pragma unroll
    for (int i = 1; i <= radius; i++)
    {
        // out_tmp += d2coef<radius>(i) * (smem[tx + i + radius] + smem[tx - i + radius]);
        out_tmp += d2coef<radius>(i) * (smem[tx + i + radius] + smem[tx - i + radius]);
        if (tx == 1)
        {
            printf("[gpu] i: %d, coeff: %e, input_plus: %e, input_minus: %e, out_tmp: %e\n", i, d2coef<radius>(i), smem[tx + i + radius], smem[tx - i + radius], out_tmp);
        }
    }
    output[tx] = out_tmp;
}

template <typename T, int radius>
void deriv0_cpu(T const *left_halo,
                T const *right_halo,
                T const *center,
                T *output,
                int n0)
{
    // Store left, right and center into one input array
    int ntotal = n0 + 2 * radius;
    T *input = (T *)malloc(ntotal * sizeof(T));
    for (int i = 0; i < radius; i++)
    {
        input[i] = left_halo[i];
        input[radius + n0 + i] = right_halo[i];
    }
    for (int i = 0; i < n0; i++)
        input[i + radius] = center[i];

    // QC
    // for (int i = 0; i < n0 + 2 * radius; i++)
    //     printf("i: %d, input: %f\n", i, input[i]);

    // Compute output
    for (int i = 0; i < n0; i++)
    {
        output[i] = d2coef<radius>(0) * input[i + radius];
        for (int j = 1; j <= radius; j++)
        {
            // output[i] += d2coef<radius>(j) * (input[i + j + radius] + input[i - j + radius]);
            output[i] += d2coef<radius>(j) * (input[i + j + radius] + input[i - j + radius]);
            if (i == 1)
                printf("[cpu] j: %d, coeff: %e, input_plus: %e, input_minus: %e, out_tmp: %e\n", j, d2coef<radius>(j), input[i + j + radius], input[i - j + radius], output[i]);
        }
    }
}

typedef float T; // Simplify notations

// Main program
int main(int argc, char **argv)
{
    // Read parameters from command line
    parfileReader par(argc, argv);

    // Get GPU properties
    cudaDeviceProp properties;
    cudaGetDeviceProperties(&properties, 0);
    printf("GPU 0: %s, global memory: %.4f (GB), %s, Number of SMs: %d, clock rate: %d\n",
           properties.name, properties.totalGlobalMem / (1024.f * 1024.f * 1024.f),
           (properties.ECCEnabled) ? "ECC on" : "ECC off",
           properties.multiProcessorCount,
           properties.clockRate);

    // Halos
    const int radius = 4;

    // Dimensions
    // int n0 = 6;
    // int n1 = par.getInt("n1", 1);
    // int n2 = par.getInt("n2", 1);
    // size_t ntotal = (size_t)n0 * n1 * n2;
    // int nhalo = (size_t)radius * n1 * n2;
    int ntotal = 6;
    int n0 = 6;
    int n1 = 1;
    int n2 = 1;
    int nhalo = radius;

    // Computational grid
    const int tile_dim0 = 6;
    const int tile_dim1 = 1;
    const int tile_dim2 = 1;
    dim3 threads(tile_dim0, tile_dim1, 1);
    dim3 blocks((tile_dim0 + tile_dim0 - 1) / tile_dim0,
                (n1 + tile_dim1 - 1) / tile_dim1,
                (n2 + tile_dim2 - 1) / tile_dim2);

    // Print dimensions
    printf("ntotal: %d, nhalo: %d\n", ntotal, nhalo);
    printf("threads.x: %d, threads.y: %d, threads.z: %d\n", threads.x, threads.y, threads.z);
    printf("blocks.x: %d, blocks.y: %d, blocks.z: %d\n", blocks.x, blocks.y, blocks.z);

    // Allocation
    T *left_halo, *right_halo, *center, *output_cpu, *output_ldg, *output_ldgsts;
    cudaMallocManaged((void **)&left_halo, nhalo * sizeof(T));
    cudaMallocManaged((void **)&right_halo, nhalo * sizeof(T));
    cudaMallocManaged((void **)&center, ntotal * sizeof(T));
    cudaMallocManaged((void **)&output_cpu, ntotal * sizeof(T));
    cudaMallocManaged((void **)&output_ldg, ntotal * sizeof(T));
    cudaMallocManaged((void **)&output_ldgsts, ntotal * sizeof(T));

    // Check that n0 >= radius

    // Initialize input
    // Randomizer rnd;
    // rnd.randomize(center, ntotal);
    // rnd.randomize(left_halo, nhalo);
    // rnd.randomize(right_halo, nhalo);
    T scale = 1.0f;
    for (int i = 0; i < radius; i++)
    {
        left_halo[i] = (T)i * scale;
        right_halo[i] = (T)(i + n0 + radius) * scale;
    }
    for (int i = 0; i < n0; i++)
        center[i] = (T)(i + radius) * scale;

    // for (int i = 0; i < radius; i++)
    //     printf("i: %d, left: %f, right: %f\n", i, left_halo[i], right_halo[i]);

    // for (int i = 0; i < n0; i++)
    //     printf("i: %d, center: %f\n", i, center[i]);

    // Initialize output
    memset(output_cpu, 0, ntotal);
    memset(output_ldg, 0, ntotal);
    memset(output_ldgsts, 0, ntotal);

    // Compute finite-difference derivative
    deriv0_cpu<T, radius>(left_halo, right_halo, center, output_cpu, ntotal);
    deriv0_ldg<T, radius, tile_dim0><<<blocks, threads>>>(left_halo, right_halo, center, output_ldg);

    // Block CPU until all kernels from default stream are executed
    cudaDeviceSynchronize();

    // Check results
    int err = 0;
    for (int i = 0; i < ntotal; i++)
    {
        if (output_cpu[i] != output_ldg[i])
        {
            printf("Error i: %d, output_cpu: %e, output_ldg: %e\n", i, output_cpu[i], output_ldg[i]);
            err++;
        }
    }
    if (!err)
        printf("Success\n");

    // Deallocation
    cudaFree(left_halo);
    cudaFree(right_halo);
    cudaFree(center);
    cudaFree(output_cpu);
    cudaFree(output_ldg);

    return 0;
}
