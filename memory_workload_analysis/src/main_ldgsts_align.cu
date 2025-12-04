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
// This sample code is modified from the original code:
//
// https://gitlab-master.nvidia.com/Devtech-Compute/ampere-perf-guide/-/blob/master/ldg_vs_ldgsts/ldgsts_align.cu
//
////////////////////////////////////////////////////////////////////////////

#include <cuda_pipeline_primitives.h>
#include "parfileReader.h"

template <typename Tload, int src_offset, int dst_offset>
__global__ void ldg_align(float *input_array, float *output_array)
{
    // 4 bytes
    if (std::is_same_v<Tload, float>)
    {
        int idx_src = src_offset + threadIdx.x;
        int idx_dst = dst_offset + threadIdx.x;
        printf("idx_src: %d, idx_dst: %d, input: %f\n", idx_src, idx_dst, input_array[idx_src]);
        output_array[idx_dst] = input_array[idx_src];
    }

    // 8 bytes
    if (std::is_same_v<Tload, float2>)
    {
        input_array += src_offset;
        output_array += dst_offset;

        int idx_src = threadIdx.x;
        int idx_dst = threadIdx.x;
        float2 *input_array_float2 = (float2 *)input_array;
        float2 *output_array_float2 = (float2 *)output_array;
        output_array_float2[idx_dst] = input_array_float2[idx_src];
    }
    // 16 bytes
    if (std::is_same_v<Tload, float4>)
    {
        input_array += src_offset;
        output_array += dst_offset;

        int idx_src = threadIdx.x;
        int idx_dst = threadIdx.x;
        float4 *input_array_float4 = (float4 *)input_array;
        float4 *output_array_float4 = (float4 *)output_array;
        output_array_float4[idx_dst] = input_array_float4[idx_src];
    }
}

template <typename Tload, int nshmem, int src_offset, int dst_offset>
__global__ void ldg_shmem_align(float *input_array, float *output_array)
{
    // 4 bytes
    if (std::is_same_v<Tload, float>)
    {
        __shared__ __align__(4) float shmem[nshmem];
        int idx_src = src_offset + threadIdx.x;
        int idx_dst = dst_offset + threadIdx.x;

        shmem[idx_dst] = input_array[idx_src];
        __syncthreads();

        output_array[idx_dst] = shmem[idx_dst];
    }

    // 8 bytes
    if (std::is_same_v<Tload, float2>)
    {
        __shared__ __align__(8) float2 shmem[nshmem];

        input_array += src_offset;
        output_array += dst_offset;
        int idx_shmem = dst_offset / 2 + threadIdx.x;
        int idx_src = threadIdx.x;
        int idx_dst = threadIdx.x;

        float2 *input_array_float2 = (float2 *)input_array;
        float2 *output_array_float2 = (float2 *)output_array;

        shmem[idx_shmem] = input_array_float2[idx_src];

        __syncthreads();

        output_array_float2[idx_dst] = shmem[idx_shmem];
    }

    // 8 bytes
    if (std::is_same_v<Tload, float4>)
    {
        __shared__ __align__(16) float4 shmem[nshmem];

        input_array += src_offset;
        output_array += dst_offset;
        int idx_shmem = dst_offset / 4 + threadIdx.x;
        int idx_src = threadIdx.x;
        int idx_dst = threadIdx.x;

        float4 *input_array_float4 = (float4 *)input_array;
        float4 *output_array_float4 = (float4 *)output_array;

        shmem[idx_shmem] = input_array_float4[idx_src];

        __syncthreads();

        output_array_float4[idx_dst] = shmem[idx_shmem];
    }
}

template <typename Tload, int nshmem, int src_offset, int dst_offset>
__global__ void ldgsts_shmem_align(float *input_array, float *output_array)
{
    // 4 bytes
    if (std::is_same_v<Tload, float>)
    {
        __shared__ __align__(4) float shmem[nshmem];
        int idx_src = src_offset + threadIdx.x;
        int idx_dst = dst_offset + threadIdx.x;

        __pipeline_memcpy_async(&shmem[idx_dst], &input_array[idx_src], sizeof(Tload));
        __pipeline_commit();
        __pipeline_wait_prior(0);
        __syncthreads();

        output_array[idx_dst] = shmem[idx_dst];
    }

    // 8 bytes
    if (std::is_same_v<Tload, float2>)
    {
        __shared__ __align__(8) float2 shmem[nshmem];

        input_array += src_offset;
        output_array += dst_offset;
        int idx_shmem = dst_offset / 2 + threadIdx.x;
        int idx_src = threadIdx.x;
        int idx_dst = threadIdx.x;

        float2 *input_array_float2 = (float2 *)input_array;
        float2 *output_array_float2 = (float2 *)output_array;

        __pipeline_memcpy_async(&shmem[idx_shmem], &input_array_float2[idx_src], sizeof(Tload));
        __pipeline_commit();
        __pipeline_wait_prior(0);
        __syncthreads();
        output_array_float2[idx_dst] = shmem[idx_shmem];
    }

    // 16 bytes
    if (std::is_same_v<Tload, float4>)
    {
        __shared__ __align__(16) float4 shmem[nshmem];

        input_array += src_offset;
        output_array += dst_offset;
        int idx_shmem = dst_offset / 4 + threadIdx.x;
        int idx_src = threadIdx.x;
        int idx_dst = threadIdx.x;

        float4 *input_array_float4 = (float4 *)input_array;
        float4 *output_array_float4 = (float4 *)output_array;

        __pipeline_memcpy_async(&shmem[idx_shmem], &input_array_float4[idx_src], sizeof(Tload));
        __pipeline_commit();
        __pipeline_wait_prior(0);
        __syncthreads();
        output_array_float4[idx_dst] = shmem[idx_shmem];
    }
}

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

    // Constant values
    const int max_offset = 128;
    const int max_nbytes_access = 16;

    // Size
    const int ntotal = (max_offset + max_nbytes_access * 31) / sizeof(float) + 1;
    printf("ntotal: %d\n", ntotal);

    // Allocation
    float *input, *output;
    cudaMallocManaged((void **)&input, ntotal * sizeof(float));
    cudaMallocManaged((void **)&output, ntotal * sizeof(float));

    // Initialize input
    for (int i = 0; i < ntotal; i++)
        input[i] = 1.0f * i;

    // LDG
    // memset(output, 0, ntotal * sizeof(float));
    // ldg_align<float, 2, 0><<<1, 32>>>(input, output);
    // cudaDeviceSynchronize();
    // for (int i = 0; i < ntotal; i++)
    //     printf("i: %d, output_float: %f\n", i, output[i]);

    // memset(output, 0, ntotal * sizeof(float));
    // ldg_align<float2, 4, 2><<<1, 32>>>(input, output);
    // cudaDeviceSynchronize();
    // for (int i = 0; i < ntotal; i++)
    //     printf("i: %d, output_float2: %f\n", i, output[i]);

    // memset(output, 0, ntotal * sizeof(float));
    // ldg_align<float4, 32, 16><<<1, 32>>>(input, output);
    // cudaDeviceSynchronize();
    // for (int i = 0; i < ntotal; i++)
    //     printf("i: %d, output_float4: %f\n", i, output[i]);

    // Shared Memory
    // memset(output, 0, ntotal * sizeof(float));
    // ldg_shmem_align<float, ntotal, 2, 0><<<1, 32>>>(input, output);
    // cudaDeviceSynchronize();
    // for (int i = 0; i < ntotal; i++)
    //     printf("i: %d, ldg_shmem_align_float: %f\n", i, output[i]);

    // memset(output, 0, ntotal * sizeof(float));
    // ldg_shmem_align<float2, ntotal, 4, 2><<<1, 32>>>(input, output);
    // cudaDeviceSynchronize();
    // for (int i = 0; i < ntotal; i++)
    //     printf("i: %d, ldg_shmem_align_float2: %f\n", i, output[i]);

    // memset(output, 0, ntotal * sizeof(float));
    // ldg_shmem_align<float4, ntotal, 32, 16><<<1, 32>>>(input, output);
    // cudaDeviceSynchronize();
    // for (int i = 0; i < ntotal; i++)
    //     printf("i: %d, ldg_shmem_align_float4: %f\n", i, output[i]);

    // LDGSTS
    memset(output, 0, ntotal * sizeof(float));
    ldgsts_shmem_align<float, ntotal, 4, 0><<<1, 32>>>(input, output);
    cudaDeviceSynchronize();
    for (int i = 0; i < ntotal; i++)
        printf("i: %d, ldgsts_shmem_align_float: %f\n", i, output[i]);

    memset(output, 0, ntotal * sizeof(float));
    ldgsts_shmem_align<float2, ntotal, 4, 0><<<1, 32>>>(input, output);
    cudaDeviceSynchronize();
    for (int i = 0; i < ntotal; i++)
        printf("i: %d, ldgsts_shmem_align_float2: %f\n", i, output[i]);

    memset(output, 0, ntotal * sizeof(float));
    ldgsts_shmem_align<float4, ntotal, 4, 0><<<1, 32>>>(input, output);
    cudaDeviceSynchronize();
    for (int i = 0; i < ntotal; i++)
        printf("i: %d, ldgsts_shmem_align_float4: %f\n", i, output[i]);        

    // Deallocation
    cudaFree(input);
    cudaFree(output);

    return 0;
}
