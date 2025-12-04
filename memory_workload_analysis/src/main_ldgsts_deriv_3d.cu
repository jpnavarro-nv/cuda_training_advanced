////////////////////////////////////////////////////////////////////////////
// Copyright (C) 2025 NVIDIA Corporation. All rights reserved.
//
// NVIDIA Sample Code
//
// Please refer to the NVIDIA end user license agreement (EULA) associated
// with this source code for terms and conditions that govern your use of
// this software. Any use, reproduction, disclosure, or distribution of
// this software and related documentation outside the terms of the EULA
// is strictly prohibited.
////////////////////////////////////////////////////////////////////////////

#include <stdio.h>
#include <stdint.h>
#include "parfileReader.h"
#include <cuda_pipeline_primitives.h>
#include "helpers.h"
#include "fd_coeffs.h"

// Function used to make sure the results are consistent across the difference kernels
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

/* 
Kernel dscription:
- Loads center and halos into shared memory without using LDGSTS for a fixed i2 value (i2 is the index on the slow direction). 
- If the threads belongs to the part of the center region adjacent to the borders, it must load the halos from either the 'left' of 'right' input arrays. Otherwise, it loads its halo directly from the center array.
- Performs a finite-difference derivative in the second fastest dimension.
- Iterates this procedure by stepping along the slow direction.
*/
template <typename T, int radius, int tile_dim0, int tile_dim1, int tile_dim2>
__global__ void deriv_ldg(T const *__restrict__ left_halo,
                          T const *__restrict__ right_halo,
                          T const *__restrict__ center,
                          T *__restrict__ output,
                          int n0, int n1, int n2)
{
    // Check there are enough points to load halos into shared memory
    static_assert(tile_dim1 >= radius);

    // Declare shared memory array, size is known at compile time
    __shared__ T smem[2 * radius + tile_dim1][tile_dim0];

    // Get thread index,
    // No need to check bounds (we make sure the dimensions are multiples of block size to simplify things)
    const int t0 = threadIdx.x;                 // Local index (inside the thread block)
    const int t1 = threadIdx.y;                 // Local index (inside the thread block)
    const int i0 = blockIdx.x * tile_dim0 + t0; // dim0 index on the main grid
    const int i1 = blockIdx.y * tile_dim1 + t1; // dim1 index on the main grid
    const int i2_start = blockIdx.z * tile_dim2;
    const int n01 = n0 * n1;

    if (i0 >= n0 || i1 >= n1)
        return;

    // Shift array pointers in dim2
    left_halo += i2_start * radius * n0;
    right_halo += i2_start * radius * n0;
    center += static_cast<size_t>(i2_start) * n01;
    output += static_cast<size_t>(i2_start) * n01;

    // Compute input/output indices
    int gind = i1 * n0 + i0;
    int gind_left = (blockIdx.y == 0) ? t1 * n0 + i0 : (i1 - radius) * n0 + i0;
    bool last_block_dim1 = (i1 + tile_dim1 >= n1) ? true : false;
    int gind_right = last_block_dim1 ? (radius - tile_dim1 + t1) * n0 + i0 : (i1 + radius) * n0 + i0;
    int stride_left = blockIdx.y == 0 ? radius * n0 : n01;
    int stride_right = last_block_dim1 ? radius * n0 : n01;

    // Loop over dim2
#pragma unroll 1
    for (int i2 = 0; i2 < tile_dim2; i2++)
    {
        // Load left halo from left
        if (t1 < radius && blockIdx.y == 0)
            smem[t1][t0] = left_halo[gind_left];

        // Load left halo from center
        else if (t1 < radius && blockIdx.y > 0)
            smem[t1][t0] = center[gind_left];

        // Load right halo from right
        if (t1 >= tile_dim1 - radius && last_block_dim1)
            smem[2 * radius + t1][t0] = right_halo[gind_right];

        // Load right halo from center
        if (t1 >= tile_dim1 - radius && !last_block_dim1)
            smem[2 * radius + t1][t0] = center[gind_right];

        // Load center
        smem[t1 + radius][t0] = center[gind];

        // Wait until all threads within block are done writing to shared memory
        __syncthreads();

        // Compute derivative
        T out_tmp = d2coef<radius>(0) * smem[t1 + radius][t0];
#pragma unroll
        for (int i = 1; i <= radius; i++)
            out_tmp += d2coef<radius>(i) * (smem[t1 + i + radius][t0] + smem[t1 - i + radius][t0]);

        // Write to global memory
        output[gind] = out_tmp;

        // Step in dim2
        gind_left += stride_left;
        gind_right += stride_right;
        gind += n01;

        // Protect shared memory array until every threads are done computing the derivative
        __syncthreads();
    }
}

/* 
Kernel dscription: 
this is the same kernel as 'deriv_ldg' but the conditional statements to check whether a thread must load its halo from left/right or center is handled more efficiently. This implementation is more optimal because it allows the compiler to fire the LDG instructions all back to back, and does not interleave them between STS instructions. This increase the Instruction-Level Parallelism (ILP) of the code, and the 'bytes in flight'. 
*/
template <typename T, int radius, int tile_dim0, int tile_dim1, int tile_dim2>
__global__ void deriv_ldg_no_branch(T const *__restrict__ left_halo,
                                    T const *__restrict__ right_halo,
                                    T const *__restrict__ center,
                                    T *__restrict__ output,
                                    int n0, int n1, int n2)
{
    // Check there are enough points to load halos into shared memory
    static_assert(tile_dim1 >= radius);

    // Declare shared memory array, size is known at compile time
    __shared__ T smem[2 * radius + tile_dim1][tile_dim0];

    // Get thread index,
    // No need to check bounds (we make sure the dimensions are multiples of block size to simplify things)
    const int t0 = threadIdx.x;                 // Local index (inside the thread block)
    const int t1 = threadIdx.y;                 // Local index (inside the thread block)
    const int i0 = blockIdx.x * tile_dim0 + t0; // dim0 index on the main grid
    const int i1 = blockIdx.y * tile_dim1 + t1; // dim1 index on the main grid
    const int i2_start = blockIdx.z * tile_dim2;
    const int n01 = n0 * n1;

    if (i0 >= n0 || i1 >= n1)
        return;

    // Shift array pointers in dim2
    left_halo += i2_start * radius * n0;
    right_halo += i2_start * radius * n0;
    center += static_cast<size_t>(i2_start) * n01;
    output += static_cast<size_t>(i2_start) * n01;

    // Compute input/output indices
    int gind = i1 * n0 + i0;
    int gind_left = (blockIdx.y == 0) ? t1 * n0 + i0 : (i1 - radius) * n0 + i0;
    bool last_block_dim1 = (i1 + tile_dim1 >= n1) ? true : false;
    int gind_right = last_block_dim1 ? (radius - tile_dim1 + t1) * n0 + i0 : (i1 + radius) * n0 + i0;
    int stride_left = blockIdx.y == 0 ? radius * n0 : n01;
    int stride_right = last_block_dim1 ? radius * n0 : n01;

    // Loop over dim2
#pragma unroll 1
    for (int i2 = 0; i2 < tile_dim2; i2++)
    {
        // This is the part that differs from 'deriv_ldg'. Observe how these conditional statements are translated into the SASS instructions. deriv_ldg
        if (t1 < radius)
        {
            auto src = (blockIdx.y == 0) ? &left_halo[gind_left] : &center[gind_left];
            smem[t1][t0] = *src;
        }

        if (t1 >= tile_dim1 - radius)
        {
            auto src = (last_block_dim1) ? &right_halo[gind_right] : &center[gind_right];
            smem[2 * radius + t1][t0] = *src;
        }

        // Load center
        smem[t1 + radius][t0] = center[gind];

        // Wait until all threads within block are done writing to shared memory
        __syncthreads();

        // Compute derivative
        T out_tmp = d2coef<radius>(0) * smem[t1 + radius][t0];
#pragma unroll
        for (int i = 1; i <= radius; i++)
            out_tmp += d2coef<radius>(i) * (smem[t1 + i + radius][t0] + smem[t1 - i + radius][t0]);

        // Write to global memory
        output[gind] = out_tmp;

        // Step in dim2
        gind_left += stride_left;
        gind_right += stride_right;
        gind += n01;

        // Protect shared memory array until every threads are done computing the derivative
        __syncthreads();
    }
}

// LDGSTS
template <typename T, int radius, int tile_dim0, int tile_dim1, int tile_dim2>
__global__ void deriv_ldgsts(T const *__restrict__ left_halo,
                             T const *__restrict__ right_halo,
                             T const *__restrict__ center,
                             T *__restrict__ output,
                             int n0, int n1, int n2)
{
    // Check there are enough points to load halos into shared memory
    static_assert(tile_dim1 >= radius);

    // Declare shared memory array, size is known at compile time
    __shared__ __align__(16) T smem[2 * radius + tile_dim1][tile_dim0];

    // Get thread index,
    // No need to check bounds (we make sure the dimensions are multiples of block size to simplify things)
    const int t0 = threadIdx.x;                 // Local index (inside the thread block)
    const int t1 = threadIdx.y;                 // Local index (inside the thread block)
    const int i0 = blockIdx.x * tile_dim0 + t0; // dim0 index on the main grid
    const int i1 = blockIdx.y * tile_dim1 + t1; // dim1 index on the main grid
    const int i2_start = blockIdx.z * tile_dim2;
    const int n01 = n0 * n1;

    if (i0 >= n0 || i1 >= n1)
        return;

    // Shift array pointers in dim2
    left_halo += i2_start * radius * n0;
    right_halo += i2_start * radius * n0;
    center += static_cast<size_t>(i2_start) * n01;
    output += static_cast<size_t>(i2_start) * n01;

    // Compute input/output indices
    int gind = i1 * n0 + i0;
    int gind_left = (blockIdx.y == 0) ? t1 * n0 + i0 : (i1 - radius) * n0 + i0;
    bool last_block_dim1 = (i1 + tile_dim1 >= n1) ? true : false;
    int gind_right = last_block_dim1 ? (radius - tile_dim1 + t1) * n0 + i0 : (i1 + radius) * n0 + i0;
    int stride_left = blockIdx.y == 0 ? radius * n0 : n01;
    int stride_right = last_block_dim1 ? radius * n0 : n01;

    // Loop over dim2
    for (int i2 = 0; i2 < tile_dim2; i2++)
    {
        // Load left halo from left
        if (t1 < radius && blockIdx.y == 0)
            __pipeline_memcpy_async(&smem[t1][t0], &left_halo[gind_left], sizeof(T));

        // Load left halo from center
        else if (t1 < radius && blockIdx.y > 0)
            __pipeline_memcpy_async(&smem[t1][t0], &center[gind_left], sizeof(T));

        // Load right halo from right
        if (t1 >= tile_dim1 - radius && last_block_dim1)
            __pipeline_memcpy_async(&smem[2 * radius + t1][t0], &right_halo[gind_right], sizeof(T));

        // Load right halo from center
        if (t1 >= tile_dim1 - radius && !last_block_dim1)
            __pipeline_memcpy_async(&smem[2 * radius + t1][t0], &center[gind_right], sizeof(T));

        // Load center
        __pipeline_memcpy_async(&smem[t1 + radius][t0], &center[gind], sizeof(T));
        __pipeline_commit();

        // Thread waits until asynchronous copy is completed
        __pipeline_wait_prior(0);

        // Wait until all threads within block are done writing to shared memory
        __syncthreads();

        // Compute derivative
        T out_tmp = d2coef<radius>(0) * smem[t1 + radius][t0];
#pragma unroll
        for (int i = 1; i <= radius; i++)
            out_tmp += d2coef<radius>(i) * (smem[t1 + i + radius][t0] + smem[t1 - i + radius][t0]);

        // Write to global memory
        output[gind] = out_tmp;

        // Step in dim2
        gind_left += stride_left;
        gind_right += stride_right;
        gind += n01;

        // Protect shared memory array until every threads are done computing the derivative
        __syncthreads();
    }
}

// LDGSTS no branch
template <typename T, int radius, int tile_dim0, int tile_dim1, int tile_dim2>
__global__ void deriv_ldgsts_no_branch(T const *__restrict__ left_halo,
                                       T const *__restrict__ right_halo,
                                       T const *__restrict__ center,
                                       T *__restrict__ output,
                                       int n0, int n1, int n2)
{
    // Check there are enough points to load halos into shared memory
    static_assert(tile_dim1 >= radius);

    // Declare shared memory array, size is known at compile time
    __shared__ __align__(16) T smem[2 * radius + tile_dim1][tile_dim0];

    // Get thread index,
    // No need to check bounds (we make sure the dimensions are multiples of block size to simplify things)
    const int t0 = threadIdx.x;                 // Local index (inside the thread block)
    const int t1 = threadIdx.y;                 // Local index (inside the thread block)
    const int i0 = blockIdx.x * tile_dim0 + t0; // dim0 index on the main grid
    const int i1 = blockIdx.y * tile_dim1 + t1; // dim1 index on the main grid
    const int i2_start = blockIdx.z * tile_dim2;
    const int n01 = n0 * n1;

    if (i0 >= n0 || i1 >= n1)
        return;

    // Shift array pointers in dim2
    left_halo += i2_start * radius * n0;
    right_halo += i2_start * radius * n0;
    center += static_cast<size_t>(i2_start) * n01;
    output += static_cast<size_t>(i2_start) * n01;

    // Compute input/output indices
    int gind = i1 * n0 + i0;
    int gind_left = (blockIdx.y == 0) ? t1 * n0 + i0 : (i1 - radius) * n0 + i0;
    bool last_block_dim1 = (i1 + tile_dim1 >= n1) ? true : false;
    int gind_right = last_block_dim1 ? (radius - tile_dim1 + t1) * n0 + i0 : (i1 + radius) * n0 + i0;
    int stride_left = blockIdx.y == 0 ? radius * n0 : n01;
    int stride_right = last_block_dim1 ? radius * n0 : n01;

    // Loop over dim2
    for (int i2 = 0; i2 < tile_dim2; i2++)
    {
        // Load left halo
        if (t1 < radius)
        {
            auto src = (blockIdx.y == 0) ? &left_halo[gind_left] : &center[gind_left];
            __pipeline_memcpy_async(&smem[t1][t0], src, sizeof(T));
        }

        // Load right halo
        if (t1 >= tile_dim1 - radius)
        {
            auto src = (last_block_dim1) ? &right_halo[gind_right] : &center[gind_right];
            __pipeline_memcpy_async(&smem[2 * radius + t1][t0], src, sizeof(T));
        }

        // Load center
        __pipeline_memcpy_async(&smem[t1 + radius][t0], &center[gind], sizeof(T));
        __pipeline_commit();

        // Thread waits until asynchronous copy is completed
        __pipeline_wait_prior(0);

        // Wait until all threads within block are done writing to shared memory
        __syncthreads();

        // Compute derivative
        T out_tmp = d2coef<radius>(0) * smem[t1 + radius][t0];
#pragma unroll
        for (int i = 1; i <= radius; i++)
            out_tmp += d2coef<radius>(i) * (smem[t1 + i + radius][t0] + smem[t1 - i + radius][t0]);

        // Write to global memory
        output[gind] = out_tmp;

        // Step in dim2
        gind_left += stride_left;
        gind_right += stride_right;
        gind += n01;

        // Protect shared memory array until every threads are done computing the derivative
        __syncthreads();
    }
}

// LDGSTS using float4 (16-byte access)
template <typename T, typename Tload, int radius, int tile_dim0, int tile_dim1, int tile_dim2>
__global__ void deriv_ldgsts_float4(T const *__restrict__ left_halo,
                                     T const *__restrict__ right_halo,
                                     T const *__restrict__ center,
                                     T *__restrict__ output,
                                     int n0, int n1, int n2)
{
    // Check there are enough points to load halos into shared memory
    static_assert(tile_dim0 == 32 && tile_dim1 == 16);
    static_assert(tile_dim0 * tile_dim1 >= tile_dim0 * (2 * radius + tile_dim1) / 4);
    static_assert((radius * sizeof(T)) % 16 == 0);

    // Declare shared memory array, size is known at compile time
    __shared__ __align__(16) T smem[2 * radius + tile_dim1][tile_dim0];

    // Get thread index,
    // No need to check bounds (we make sure the dimensions are multiples of block size to simplify things)
    const int t0 = threadIdx.x;                 // Local index (inside the thread block)
    const int t1 = threadIdx.y;                 // Local index (inside the thread block)
    const int i0 = blockIdx.x * tile_dim0 + t0; // dim0 index on the main grid
    const int i1 = blockIdx.y * tile_dim1 + t1; // dim1 index on the main grid
    const int i2_start = blockIdx.z * tile_dim2;
    const int n01 = n0 * n1;

    if (i0 >= n0 || i1 >= n1)
        return;

    // Shift array pointers in dim2
    left_halo += i2_start * radius * n0;
    right_halo += i2_start * radius * n0;
    center += static_cast<size_t>(i2_start) * n01;
    output += static_cast<size_t>(i2_start) * n01;

    // Reindex the threads
    // t0_load and t1_load are the shared memory indices at which we are going to store the data
    int t0_load = 4 * (threadIdx.x % (tile_dim0 / 4));
    int t1_load = 4 * threadIdx.y + threadIdx.x / (tile_dim0 / 4);

    // Subset of threads that will load data
    bool load_left = t1_load < radius;
    bool load_middle = t1_load >= radius && t1_load < radius + tile_dim1;
    bool load_right = t1_load >= radius + tile_dim1 && t1_load < 2 * radius + tile_dim1;

    // Compute output index
    int gind_out = i1 * n0 + i0;

    // Check if on edges of dim1
    bool first_block_dim1 = (blockIdx.y == 0) ? true : false;
    bool last_block_dim1 = (i1 + tile_dim1 >= n1) ? true : false;

    int i0_load = blockIdx.x * tile_dim0 + t0_load;
    int i1_load;

    int gind_load_left, gind_load_right, gind_load_center;

    // Load left halo from center array
    if (!first_block_dim1)
        i1_load = blockIdx.y * tile_dim1 + t1_load - radius; // shift by -radius

    // Load left halo from left array
    else
        i1_load = t1_load;

    gind_load_left = i1_load * n0 + i0_load;

    // Load interior from center array
    i1_load = blockIdx.y * tile_dim1 + t1_load - radius;
    gind_load_center = i1_load * n0 + i0_load;

    // Find index on right array
    if (last_block_dim1)
        i1_load = t1_load - radius - tile_dim1;

    // Gmem index to memory for right halo
    gind_load_right = i1_load * n0 + i0_load;

    // Strides
    int stride_left = first_block_dim1 ? radius * n0 : n01;
    int stride_right = last_block_dim1 ? radius * n0 : n01;

    // Loop over dim2
    for (int i2 = 0; i2 < tile_dim2; i2++)
    {
        // Load left halo
        if (load_left)
        {
            auto src = first_block_dim1 ? &left_halo[gind_load_left] : &center[gind_load_left];
            __pipeline_memcpy_async(&smem[t1_load][t0_load], src, sizeof(Tload));
        }

        // Load right halo
        if (load_right)
        {
            auto src = last_block_dim1 ? &right_halo[gind_load_right] : &center[gind_load_right];
            __pipeline_memcpy_async(&smem[t1_load][t0_load], src, sizeof(Tload));
        }

        // Load center
        if (load_middle)
        {
            __pipeline_memcpy_async(&smem[t1_load][t0_load], &center[gind_load_center], sizeof(Tload));
        }
        __pipeline_commit();

        // // Load left halo from left array
        // if (load_left && first_block_dim1)
        //     __pipeline_memcpy_async(&smem[t1_load][t0_load], &left_halo[gind_load_left], sizeof(Tload));
        // // Load left halo from center array
        // else if (load_left && !first_block_dim1)
        //     __pipeline_memcpy_async(&smem[t1_load][t0_load], &center[gind_load_left], sizeof(Tload));
        // // Load right halo from right array
        // if (load_right && last_block_dim1)
        //     __pipeline_memcpy_async(&smem[t1_load][t0_load], &right_halo[gind_load_right], sizeof(Tload));
        // // Load right halo from center arrays
        // if (load_right && !last_block_dim1)
        //     __pipeline_memcpy_async(&smem[t1_load][t0_load], &center[gind_load_right], sizeof(Tload));
        // // Load center
        // if (load_middle)
        //     __pipeline_memcpy_async(&smem[t1_load][t0_load], &center[gind_load_center], sizeof(Tload));
        // __pipeline_commit();

        // Thread waits until asynchronous copy is completed
        __pipeline_wait_prior(0);

        // Wait until all threads within block are done writing to shared memory
        __syncthreads();

        // Compute derivative
        T out_tmp = d2coef<radius>(0) * smem[t1 + radius][t0];
#pragma unroll
        for (int i = 1; i <= radius; i++)
            out_tmp += d2coef<radius>(i) * (smem[t1 + i + radius][t0] + smem[t1 - i + radius][t0]);

        // Write to global memory
        output[gind_out] = out_tmp;

        // Step in dim2
        gind_load_left += stride_left;
        gind_load_right += stride_right;
        gind_load_center += n01;
        gind_out += n01;

        // Protect shared memory array until every threads are done computing the derivative
        __syncthreads();
    }
}

// LDGSTS using float4 (16-byte access), overlap copy/compute
template <typename T, typename Tload, int nplane, int radius, int tile_dim0, int tile_dim1, int tile_dim2>
__global__ void deriv_ldgsts_overlap(T const *__restrict__ left_halo,
                                             T const *__restrict__ right_halo,
                                             T const *__restrict__ center,
                                             T *__restrict__ output,
                                             int n0, int n1, int n2)
{
    // Check there are enough points to load halos into shared memory
    static_assert(tile_dim0 == 32 && tile_dim1 == 16);
    static_assert(tile_dim0 * tile_dim1 >= tile_dim0 * (2 * radius + tile_dim1) / 4);
    static_assert((radius * sizeof(T)) % 16 == 0);

    // Declare shared memory array, size is known at compile time
    __shared__ __align__(16) T smem[nplane * 2][2 * radius + tile_dim1][tile_dim0];

    // Get thread index,
    // No need to check bounds (we make sure the dimensions are multiples of block size to simplify things)
    const int t0 = threadIdx.x;                 // Local index (inside the thread block)
    const int t1 = threadIdx.y;                 // Local index (inside the thread block)
    const int i0 = blockIdx.x * tile_dim0 + t0; // dim0 index on the main grid
    const int i1 = blockIdx.y * tile_dim1 + t1; // dim1 index on the main grid
    const int i2_start = blockIdx.z * tile_dim2;
    const int n01 = n0 * n1;

    if (i0 >= n0 || i1 >= n1)
        return;

    // Shift array pointers in dim2
    left_halo += i2_start * radius * n0;
    right_halo += i2_start * radius * n0;
    center += static_cast<size_t>(i2_start) * n01;
    output += static_cast<size_t>(i2_start) * n01;

    // Reindex the threads
    // t0_load and t1_load are the shared memory indices at which we are going to store the data
    int t0_load = 4 * (threadIdx.x % (tile_dim0 / 4));
    int t1_load = 4 * threadIdx.y + threadIdx.x / (tile_dim0 / 4);

    // Subset of threads that will load data
    bool load_left = t1_load < radius;
    bool load_middle = t1_load >= radius && t1_load < radius + tile_dim1;
    bool load_right = t1_load >= radius + tile_dim1 && t1_load < 2 * radius + tile_dim1;

    // Compute output index
    int gind_out = i1 * n0 + i0;

    // Check if on edges of dim1
    bool first_block_dim1 = (blockIdx.y == 0) ? true : false;
    bool last_block_dim1 = (i1 + tile_dim1 >= n1) ? true : false;

    int i0_load = blockIdx.x * tile_dim0 + t0_load;
    int i1_load;

    int gind_load_left, gind_load_right, gind_load_center;

    // Load left halo from center array
    if (!first_block_dim1)
        i1_load = blockIdx.y * tile_dim1 + t1_load - radius; // shift by -radius

    // Load left halo from left array
    else
        i1_load = t1_load;

    gind_load_left = i1_load * n0 + i0_load;

    // Load interior from center array
    i1_load = blockIdx.y * tile_dim1 + t1_load - radius;
    gind_load_center = i1_load * n0 + i0_load;

    // Find index on right array
    if (last_block_dim1)
        i1_load = t1_load - radius - tile_dim1;

    // Gmem index to memory for right halo
    gind_load_right = i1_load * n0 + i0_load;

    // Strides
    int stride_left = first_block_dim1 ? radius * n0 : n01;
    int stride_right = last_block_dim1 ? radius * n0 : n01;

    // Priming
    int ismem = 0;

    // Load left halo
    if (load_left)
    {
        auto src = first_block_dim1 ? &left_halo[gind_load_left] : &center[gind_load_left];
        __pipeline_memcpy_async(&smem[ismem][t1_load][t0_load], src, sizeof(Tload));
    }

    // Load right halo
    if (load_right)
    {
        auto src = last_block_dim1 ? &right_halo[gind_load_right] : &center[gind_load_right];
        __pipeline_memcpy_async(&smem[ismem][t1_load][t0_load], src, sizeof(Tload));
    }

    // Load center
    if (load_middle)
    {
        __pipeline_memcpy_async(&smem[ismem][t1_load][t0_load], &center[gind_load_center], sizeof(Tload));
    }
    __pipeline_commit();

    gind_load_left += stride_left;
    gind_load_right += stride_right;
    gind_load_center += n01;

    // Switch smem array
    ismem ^= 1;

    // Loop over dim2
    for (int i2 = 0; i2 < tile_dim2; i2++)
    {
        if (i2 < tile_dim2 - 1)
        {
            // Load left halo
            if (load_left)
            {
                auto src = first_block_dim1 ? &left_halo[gind_load_left] : &center[gind_load_left];
                __pipeline_memcpy_async(&smem[ismem][t1_load][t0_load], src, sizeof(Tload));
            }

            // Load right halo
            if (load_right)
            {
                auto src = last_block_dim1 ? &right_halo[gind_load_right] : &center[gind_load_right];
                __pipeline_memcpy_async(&smem[ismem][t1_load][t0_load], src, sizeof(Tload));
            }

            // Load center
            if (load_middle)
            {
                __pipeline_memcpy_async(&smem[ismem][t1_load][t0_load], &center[gind_load_center], sizeof(Tload));
            }
            __pipeline_commit();
        }

        // Switch smem array
        ismem ^= 1; 

        // Thread waits until asynchronous copy is completed
        if (i2 < tile_dim2 - 1)
            __pipeline_wait_prior(1);
        else
            __pipeline_wait_prior(0);

        // Wait until all threads within block are done writing to shared memory
        __syncthreads();

        // Compute derivative
        T out_tmp = d2coef<radius>(0) * smem[ismem][t1 + radius][t0];
#pragma unroll
        for (int i = 1; i <= radius; i++)
            out_tmp += d2coef<radius>(i) * (smem[ismem][t1 + i + radius][t0] + smem[ismem][t1 - i + radius][t0]);

        // Write to global memory
        output[gind_out] = out_tmp;

        // Step in dim2
        gind_load_left += stride_left;
        gind_load_right += stride_right;
        gind_load_center += n01;
        gind_out += n01;

        // Protect shared memory array until every threads are done computing the derivative
        __syncthreads();
    }
}

// CPU version for QC
template <typename T, int radius>
void deriv_cpu(T const *left_halo,
               T const *right_halo,
               T const *center,
               T *output,
               int n0, int n1, int n2)
{
    // Store left, right and center into one input array
    int n1_pad = n1 + 2 * radius;
    size_t ntotal = (size_t)n0 * n1_pad * n2;
    T *input = (T *)malloc(ntotal * sizeof(T));

    for (int i2 = 0; i2 < n2; i2++)
    {
        // Halos
        for (int i1 = 0; i1 < radius; i1++)
            for (int i0 = 0; i0 < n0; i0++)
            {
                size_t idx_halo = i2 * radius * n0 + i1 * n0 + i0;
                size_t idx0 = i2 * n1_pad * n0 + i1 * n0 + i0;
                size_t idx1 = i2 * n1_pad * n0 + (radius + n1 + i1) * n0 + i0;
                input[idx0] = left_halo[idx_halo];
                input[idx1] = right_halo[idx_halo];
            }

        // Center
        for (int i1 = 0; i1 < n1; i1++)
            for (int i0 = 0; i0 < n0; i0++)
            {
                size_t idx_in = i2 * n1 * n0 + i1 * n0 + i0;
                size_t idx_out = i2 * n1_pad * n0 + (radius + i1) * n0 + i0;
                input[idx_out] = center[idx_in];
            }
    }

    // Use input array to compute derivative
    for (int i2 = 0; i2 < n2; i2++)
        for (int i1 = 0; i1 < n1; i1++)
            for (int i0 = 0; i0 < n0; i0++)
            {
                size_t idx_in = i2 * n1_pad * n0 + (i1 + radius) * n0 + i0;
                size_t idx_out = i2 * n1 * n0 + i1 * n0 + i0;
                output[idx_out] = d2coef<radius>(0) * input[idx_in];
                for (int j = 1; j <= radius; j++)
                {
                    output[idx_out] += d2coef<radius>(j) * (input[idx_in + j * n0] + input[idx_in - j * n0]);
                }
            }
    free(input);
}

// Type
typedef float T;
typedef float4 Tload;

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

    // Block dimensions
    const int tile_dim0 = 32;
    const int tile_dim1 = 16;
    const int tile_dim2 = 2;
    const int radius = 4;

    static_assert(tile_dim0 * tile_dim1 <= 1024); // Number of threads per block cannot exceed 1024

    // Computational domain
    const int n0 = par.getInt("n0", tile_dim0);
    const int n1 = par.getInt("n1", tile_dim1);
    const int n2 = par.getInt("n2", tile_dim2);
    const size_t n_center = (size_t)n0 * n1 * n2;
    const int n_halo = radius * n0 * n2;
    if (n0 % tile_dim0 != 0 || n1 % tile_dim1 != 0 || n2 % tile_dim2 != 0)
        throw std::runtime_error("n0/n2/n2 must be a multiple of the thread block size\n");

    // Grid/blocks
    dim3 threads(tile_dim0, tile_dim1, 1);
    dim3 blocks((n0 + tile_dim0 - 1) / tile_dim0,
                (n1 + tile_dim1 - 1) / tile_dim1,
                (n2 + tile_dim2 - 1) / tile_dim2);

    // QC
    printf("n0: %d, n1: %d, n2: %d\n", n0, n1, n2);
    printf("threads.x: %d, threads.y: %d, threads.z: %d\n", threads.x, threads.y, threads.z);
    printf("blocks.x: %d, blocks.y: %d, blocks.z: %d\n", blocks.x, blocks.y, blocks.z);

    // Allocation: use unified memory to simplify code
    T *left_halo, *right_halo, *center;
    T *output_cpu, *output_gpu_baseline, *output_gpu;
    cudaMallocManaged((void **)&left_halo, n_halo * sizeof(T));
    cudaMallocManaged((void **)&right_halo, n_halo * sizeof(T));
    cudaMallocManaged((void **)&center, n_center * sizeof(T));
    cudaMallocManaged((void **)&output_cpu, n_center * sizeof(T));
    cudaMallocManaged((void **)&output_gpu_baseline, n_center * sizeof(T));
    cudaMallocManaged((void **)&output_gpu, n_center * sizeof(T));

    // Initialize input
    Randomizer<T> rnd;
    rnd.randomize(center, n_center);
    rnd.randomize(left_halo, n_halo);
    rnd.randomize(right_halo, n_halo);
    cudaDeviceSynchronize();

    // Compute finite-difference derivative
    memset(output_cpu, 0, n_center);
    deriv_cpu<T, radius>(left_halo, right_halo, center, output_cpu, n0, n1, n2);

    // LDG
    memset(output_gpu_baseline, 0, n_center);
    deriv_ldg<T, radius, tile_dim0, tile_dim1, tile_dim2><<<blocks, threads>>>(left_halo, right_halo, center, output_gpu_baseline, n0, n1, n2);
    cudaDeviceSynchronize();
    check_result<T>(output_cpu, output_gpu_baseline, n0, n1, n2, "ldg");

    // LDG no branching
    memset(output_gpu, 0, n_center);
    deriv_ldg_no_branch<T, radius, tile_dim0, tile_dim1, tile_dim2><<<blocks, threads>>>(left_halo, right_halo, center, output_gpu, n0, n1, n2);
    cudaDeviceSynchronize();
    check_result<T>(output_gpu_baseline, output_gpu, n0, n1, n2, "ldg_no_branch");

    // LDGSTS
    memset(output_gpu, 0, n_center);
    deriv_ldgsts<T, radius, tile_dim0, tile_dim1, tile_dim2><<<blocks, threads>>>(left_halo, right_halo, center, output_gpu, n0, n1, n2);
    cudaDeviceSynchronize();
    check_result<T>(output_gpu_baseline, output_gpu, n0, n1, n2, "ldgsts");

    // LDGSTS no branch
    memset(output_gpu, 0, n_center);
    deriv_ldgsts_no_branch<T, radius, tile_dim0, tile_dim1, tile_dim2><<<blocks, threads>>>(left_halo, right_halo, center, output_gpu, n0, n1, n2);
    cudaDeviceSynchronize();
    check_result<T>(output_gpu_baseline, output_gpu, n0, n1, n2, "ldgsts_no_branch");

    // LDGSTS float4
    memset(output_gpu, 0, n_center);
    deriv_ldgsts_float4<T, Tload, radius, tile_dim0, tile_dim1, tile_dim2><<<blocks, threads>>>(left_halo, right_halo, center, output_gpu, n0, n1, n2);
    cudaDeviceSynchronize();
    check_result<T>(output_gpu_baseline, output_gpu, n0, n1, n2, "ldgsts_float4");

    // LDGSTS with overlap
    memset(output_gpu, 0, n_center);
    deriv_ldgsts_overlap<T, Tload, 1, radius, tile_dim0, tile_dim1, tile_dim2><<<blocks, threads>>>(left_halo, right_halo, center, output_gpu, n0, n1, n2);
    cudaDeviceSynchronize();
    check_result<T>(output_gpu_baseline, output_gpu, n0, n1, n2, "ldgsts_overlap1");

    // LDGSTS with overlap, multiple planes in advance (i.e. > 1)
    memset(output_gpu, 0, n_center);
    deriv_ldgsts_overlap<T, Tload, 4, radius, tile_dim0, tile_dim1, tile_dim2><<<blocks, threads>>>(left_halo, right_halo, center, output_gpu, n0, n1, n2);
    cudaDeviceSynchronize();
    check_result<T>(output_gpu_baseline, output_gpu, n0, n1, n2, "ldgsts_overlap4");

    // Block CPU until all kernels from default stream are executed
    cudaDeviceSynchronize();

    // Deallocation
    cudaFree(left_halo);
    cudaFree(right_halo);
    cudaFree(center);
    cudaFree(output_cpu);
    cudaFree(output_gpu_baseline);
    cudaFree(output_gpu);

    return 0;
}
