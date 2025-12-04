////////////////////////////////////////////////////////////////////////////
//
// Copyright (C) 2025 NVIDIA Corporation. All rights reserved.
//
// NVIDIA Sample Code
//
// Please refer to the NVIDIA end user license agreement (EULA) associated
// with this source code for terms and conditions that govern your use of
// this software. Any use, reproduction, disclosure, or distribution of
// this software and related documentation outside the terms of the EULA
// is strictly prohibited.
//
////////////////////////////////////////////////////////////////////////////

#include <stdio.h>
#include <stdint.h>
#include "parfileReader.h"
#include <cuda_pipeline_primitives.h>
#include "helpers.h"
#include "fd_coeffs.h"

// QC
template <typename T>
void check_result(T *output_cpu, T *output_gpu, int n0, int n1, int n2, std::string kernel_version)
{
    size_t err = 0;
    T tol = 1e-4;
    for (int i2 = 0; i2 < n2; i2++)
        for (int i1 = 0; i1 < n1; i1++)
            for (int i0 = 0; i0 < n0; i0++)
            {
                size_t idx = (size_t)i2 * n1 * n0 + i1 * n0 + i0;
                T rel_err = abs((output_gpu[idx] - output_cpu[idx]) / output_cpu[idx]);
                if (rel_err > tol)
                {
                    printf("Error %s i0: %d, i1: %d, i2: %d, output_cpu : %e, output_gpu: %e, relative err: %e\n ",
                           kernel_version.c_str(),
                           i0, i1, i2,
                           output_cpu[idx],
                           output_gpu[idx],
                           rel_err);

                    err++;
                }
            }
    if (!err)
        printf("Success %s\n", kernel_version.c_str());
}

// LDGs are issued back to back
template <typename T, int radius, int tile_dim0>
__global__ void deriv1d_ldg(T const *__restrict__ left_halo,
                            T const *__restrict__ right_halo,
                            T const *__restrict__ center,
                            T *__restrict__ output)
{
    // Same threads can potentially load both halos
    static_assert(tile_dim0 >= radius);

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

    // Compute derivative and write output to global memory
    T out_tmp = d2coef<radius>(0) * smem[tx + radius];
#pragma unroll
    for (int i = 1; i <= radius; i++)
        out_tmp += d2coef<radius>(i) * (smem[tx + i + radius] + smem[tx - i + radius]);

    output[tx] = out_tmp;
}

// LDG and STS instructions are interleaved (caused by the "else if")
template <typename T, int radius, int tile_dim0>
__global__ void deriv1d_ldg_sts(T const *__restrict__ left_halo,
                                T const *__restrict__ right_halo,
                                T const *__restrict__ center,
                                T *__restrict__ output)
{
    // Same threads cannot load both halos, which is why at least 2 * radius are needed
    static_assert(tile_dim0 >= 2 * radius);

    // Declare shared memory array, size is known at compile time
    __shared__ T smem[2 * radius + tile_dim0];

    // Get thread index
    const int tx = threadIdx.x;
    if (tx >= tile_dim0)
        return;

    // Fill temporary shared memory input array
    if (tx < radius)
        smem[tx] = left_halo[tx];

    else if (tx >= tile_dim0 - radius)
        smem[2 * radius + tx] = right_halo[radius - tile_dim0 + tx];

    smem[tx + radius] = center[tx];

    // Wait until all threads within block are done writing to shared memory
    __syncthreads();

    // Compute derivative and write output to global memory
    T out_tmp = d2coef<radius>(0) * smem[tx + radius];
#pragma unroll
    for (int i = 1; i <= radius; i++)
        out_tmp += d2coef<radius>(i) * (smem[tx + i + radius] + smem[tx - i + radius]);

    output[tx] = out_tmp;
}

// No branching, reduces the number of instructions
template <typename T, int radius, int tile_dim0>
__global__ void deriv1d_ldg_no_branch(T const *__restrict__ left_halo,
                                      T const *__restrict__ right_halo,
                                      T const *__restrict__ center,
                                      T *__restrict__ output)
{
    // Same threads cannot load both halos, which is why at least 2 * radius are needed
    static_assert(tile_dim0 >= 2 * radius);

    // Declare shared memory array, size is known at compile time
    __shared__ T smem[2 * radius + tile_dim0];

    // Get thread index
    const int tx = threadIdx.x;
    if (tx >= tile_dim0)
        return;

    // Only one instruction is issued by the warp
    if (tx < 2 * radius)
    {
        auto src = (left_halo + (right_halo - radius - left_halo) * (tx / radius))[tx]
        auto src = (tx < radius) ? &left_halo[tx] : &right_halo[tx - radius];
        auto dst = (tx < radius) ? &smem[tx] : &smem[tx + tile_dim0];
        *dst = *src;
    }

    smem[tx + radius] = center[tx];

    // Wait until all threads within block are done writing to shared memory
    __syncthreads();

    // Compute derivative and write output to global memory
    T out_tmp = d2coef<radius>(0) * smem[tx + radius];
#pragma unroll
    for (int i = 1; i <= radius; i++)
        out_tmp += d2coef<radius>(i) * (smem[tx + i + radius] + smem[tx - i + radius]);

    output[tx] = out_tmp;
}

// LDGSTS
template <typename T, int radius, int tile_dim0>
__global__ void deriv1d_ldgsts(T const *__restrict__ left_halo,
                               T const *__restrict__ right_halo,
                               T const *__restrict__ center,
                               T *__restrict__ output)
{
    // Same threads can potentially load both halos
    static_assert(tile_dim0 >= radius);

    // Declare shared memory array, size is known at compile time
    __shared__ __align__(16) T smem[2 * radius + tile_dim0];

    // Get thread index
    const int tx = threadIdx.x;
    if (tx >= tile_dim0)
        return;

    // Fill temporary shared memory input array
    if (tx < radius)
        __pipeline_memcpy_async(&smem[tx], &left_halo[tx], sizeof(T));

    if (tx >= tile_dim0 - radius)
        __pipeline_memcpy_async(&smem[2 * radius + tx], &right_halo[radius - tile_dim0 + tx], sizeof(T));

    __pipeline_memcpy_async(&smem[tx + radius], &center[tx], sizeof(T));
    __pipeline_commit();

    // Thread waits until asynchronous copy is completed
    __pipeline_wait_prior(0);

    // Wait until all threads within block are done writing to shared memory
    __syncthreads();

    // Compute derivative and write output to global memory
    T out_tmp = d2coef<radius>(0) * smem[tx + radius];
#pragma unroll
    for (int i = 1; i <= radius; i++)
        out_tmp += d2coef<radius>(i) * (smem[tx + i + radius] + smem[tx - i + radius]);

    output[tx] = out_tmp;
}

// LDGSTS without branching, reduced number of instructions
template <typename T, int radius, int tile_dim0>
__global__ void deriv1d_ldgsts_no_branch(T const *__restrict__ left_halo,
                                         T const *__restrict__ right_halo,
                                         T const *__restrict__ center,
                                         T *__restrict__ output)
{
    // Same threads cannot load both halos, which is why at least 2 * radius are needed
    static_assert(tile_dim0 >= 2 * radius);

    // Declare shared memory array, size is known at compile time
    __shared__ __align__(16) T smem[2 * radius + tile_dim0];

    // Get thread index
    const int tx = threadIdx.x;
    if (tx >= tile_dim0)
        return;

    // Only one instruction is issued by the warp
    if (tx < 2 * radius)
    {
        auto src = (tx < radius) ? &left_halo[tx] : &right_halo[tx - radius];
        auto dst = (tx < radius) ? &smem[tx] : &smem[tx + tile_dim0];
        __pipeline_memcpy_async(dst, src, sizeof(T));
    }

    __pipeline_memcpy_async(&smem[tx + radius], &center[tx], sizeof(T));
    __pipeline_commit();

    // Thread waits until asynchronous copy is completed
    __pipeline_wait_prior(0);

    // Wait until all threads within block are done writing to shared memory
    __syncthreads();

    // Compute derivative and write output to global memory
    T out_tmp = d2coef<radius>(0) * smem[tx + radius];
#pragma unroll
    for (int i = 1; i <= radius; i++)
        out_tmp += d2coef<radius>(i) * (smem[tx + i + radius] + smem[tx - i + radius]);

    output[tx] = out_tmp;
}

template <typename T, int radius>
void deriv1d_cpu(T const *left_halo,
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

    // Compute output
    for (int i = 0; i < n0; i++)
    {
        output[i] = d2coef<radius>(0) * input[i + radius];
        for (int j = 1; j <= radius; j++)
            output[i] += d2coef<radius>(j) * (input[i + j + radius] + input[i - j + radius]);
    }

    free(input);
}

// Type
typedef float T;

// Main program to illustrate the use of LDGSTS for finite-difference operators
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

    // Dimensions
    const int tile_dim0 = 32; // Launch only one warp
    const int radius = 2;

    // Inputs: left, right and center
    // left and right could typically correspond to adjacent domains with different physics or PML layers
    T *left_halo, *right_halo, *center; 
    T *output_cpu, *output_gpu;
    cudaMallocManaged((void **)&left_halo, radius * sizeof(T));
    cudaMallocManaged((void **)&right_halo, radius * sizeof(T));
    cudaMallocManaged((void **)&center, tile_dim0 * sizeof(T));
    cudaMallocManaged((void **)&output_cpu, tile_dim0 * sizeof(T));
    cudaMallocManaged((void **)&output_gpu, tile_dim0 * sizeof(T));

    // Initialize input
    Randomizer<T> rnd;
    rnd.randomize(center, tile_dim0);
    rnd.randomize(left_halo, radius);
    rnd.randomize(right_halo, radius);
    cudaDeviceSynchronize();

    // Other way to initialize input arrays
    // T scale = 1.0f;
    // for (int i = 0; i < radius; i++)
    // {
    //     left_halo[i] = (T)i * scale;
    //     right_halo[i] = (T)(i + tile_dim0 + radius) * scale;
    // }
    // for (int i = 0; i < tile_dim0; i++)
    //     center[i] = (T)(i + radius) * scale;

    /* 
    Code description:
    All kernels compute a finite-difference derivative at every point of the center array, which require some values outside of this array located in the halos (left and right).
    */

    // CPU Baseline to make sure results are identical CPU/GPU
    memset(output_cpu, 0, tile_dim0);
    deriv1d_cpu<T, radius>(left_halo, right_halo, center, output_cpu, tile_dim0);

    // LDG: this kernel issues three back-to-back LDG instructions followed by three back-to-back STS instructions
    memset(output_gpu, 0, tile_dim0);
    deriv1d_ldg<T, radius, tile_dim0><<<1, tile_dim0>>>(left_halo, right_halo, center, output_gpu);
    cudaDeviceSynchronize();
    check_result<T>(output_cpu, output_gpu, tile_dim0, 1, 1, "ldg");

    // LDG/STS interleaved
    // This kernel issues three LDG/STS, not three LDG followed by three STS.
    // This kernel is less optimal than "deriv1d_ldg": why?
    memset(output_gpu, 0, tile_dim0);
    deriv1d_ldg_sts<T, radius, tile_dim0><<<1, tile_dim0>>>(left_halo, right_halo, center, output_gpu);
    cudaDeviceSynchronize();
    check_result<T>(output_cpu, output_gpu, tile_dim0, 1, 1, "ldg_sts");

    // LDG with branching removed
    memset(output_gpu, 0, tile_dim0);
    deriv1d_ldg_no_branch<T, radius, tile_dim0><<<1, tile_dim0>>>(left_halo, right_halo, center, output_gpu);
    cudaDeviceSynchronize();
    check_result<T>(output_cpu, output_gpu, tile_dim0, 1, 1, "ldg_no_branch");

    // LDGSTS 
    memset(output_gpu, 0, tile_dim0);
    deriv1d_ldgsts<T, radius, tile_dim0><<<1, tile_dim0>>>(left_halo, right_halo, center, output_gpu);
    cudaDeviceSynchronize();
    check_result<T>(output_cpu, output_gpu, tile_dim0, 1, 1, "ldgsts");

    // LDGSTS, branching removed 
    memset(output_gpu, 0, tile_dim0);
    deriv1d_ldgsts_no_branch<T, radius, tile_dim0><<<1, tile_dim0>>>(left_halo, right_halo, center, output_gpu);
    cudaDeviceSynchronize();
    check_result<T>(output_cpu, output_gpu, tile_dim0, 1, 1, "ldgsts_no_branch");

    // Deallocation
    cudaFree(left_halo);
    cudaFree(right_halo);
    cudaFree(center);
    cudaFree(output_cpu);
    cudaFree(output_gpu);

    return 0;
}
