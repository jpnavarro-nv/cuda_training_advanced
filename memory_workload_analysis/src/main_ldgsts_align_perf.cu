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
// Original authors are Vishal Mehta and Guillaume Thomas-Collignon
////////////////////////////////////////////////////////////////////////////

#include <cuda_pipeline_primitives.h>
#include "parfileReader.h"

// Kernel to test LDGSTS performance
template <int nchunk, int block_size, int nbytes, size_t n_shmem>
__global__ void ldgsts_align(char *input_array, int gmem_offset, int shmem_offset, int niter, int *compute_time)
{
    // extern __shared__ char shmem[];
    __shared__ char shmem[n_shmem];
    uint32_t timer = -clock();

#pragma unroll 1
    for (int i = 0; i < niter; i++)
    {
#pragma unroll
        for (int ichunk = 0; ichunk < nchunk; ichunk++)
        {
            size_t idx_src = (size_t)i * nchunk * block_size * nbytes + ichunk * block_size * nbytes + threadIdx.x * nbytes + gmem_offset;
            size_t idx_dst = (size_t)ichunk * block_size * nbytes + threadIdx.x * nbytes + shmem_offset;
            __pipeline_memcpy_async(&shmem[idx_dst], &input_array[idx_src], nbytes);
        }
        __pipeline_commit();
        __pipeline_wait_prior(0);
    }

    __syncthreads();

    timer += clock();
    compute_time[blockIdx.x] = timer;
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
    const int nbytes_access = 16;
    const int grid_size = properties.multiProcessorCount;
    const int block_size = 256;
    const int nchunk = 8;
    const int niter_max = 100;

    // gmem offsets
    int nval_gmem = par.getInt("nval_gmem");
    int *gmem_offsets = (int *)malloc(nval_gmem * sizeof(int));
    for (int i = 0; i < nval_gmem; i++)
        gmem_offsets[i] = i * nbytes_access;

    // Shmem offsets
    int nval_shmem = par.getInt("nval_shmem");
    int *shmem_offsets = (int *)malloc(nval_shmem * sizeof(int));
    for (int i = 0; i < nval_shmem; i++)
        shmem_offsets[i] = i * nbytes_access;

    // Read parameters from command line
    int niter = par.getInt("niter", 50);

    // Check parameters' validity
    if (niter >= niter_max)
        throw std::runtime_error("Error, niter too large\n");

    // Size
    const size_t n_shmem = max_offset + (block_size - 1) * nbytes_access + block_size * nbytes_access * (nchunk - 1);
    size_t n_gmem = max_offset + (block_size - 1) * nbytes_access + block_size * nbytes_access * (nchunk - 1) + block_size * nbytes_access * nchunk * (niter_max - 1);
    size_t n_data = nbytes_access * block_size * nchunk * niter_max;

    // Allocation
    char *input_array;
    char *output_array;
    int *compute_time;
    cudaMallocManaged((void **)&input_array, n_gmem * sizeof(char));
    cudaMallocManaged((void **)&output_array, n_shmem * sizeof(char));
    cudaMallocManaged((void **)&compute_time, grid_size * sizeof(int));

    // QC
    printf("n_shmem: %lu, n_gmem: %lu, original: %d, n_data: %lu\n", n_shmem, n_gmem, 16 * 1024 * 1024, n_data);
    printf("block_size: %d, grid_size: %d\n", block_size, grid_size);

    // Warmup
    for (int i = 0; i < 100; i++)
    {
        memset(compute_time, 0, grid_size * sizeof(int));
        ldgsts_align<nchunk, block_size, nbytes_access, n_shmem><<<grid_size, block_size>>>(input_array, 0, 0, niter, compute_time);
        cudaDeviceSynchronize();
    }

    // Performance
    for (int igmem = 0; igmem < nval_gmem; igmem++)
    {
        int gmem_offset = gmem_offsets[igmem];
        for (int ishmem = 0; ishmem < nval_shmem; ishmem++)
        {
            memset(compute_time, 0, grid_size * sizeof(int));
            int shmem_offset = shmem_offsets[ishmem];
            ldgsts_align<nchunk, block_size, nbytes_access, n_shmem><<<grid_size, block_size>>>(input_array, gmem_offset, shmem_offset, niter, compute_time);
            cudaDeviceSynchronize();
            size_t total = 0;
            for (int i = 0; i < grid_size; i++)
                total += compute_time[i];
            // printf("gmem_offset: %d, shmem_offset: %d, total: %lu\n", gmem_offset, shmem_offset, total);

            // Write perf to file
            std::ofstream outputFile_results;
            std::string filename_results = par.getString("perf_results");
            outputFile_results.open(filename_results, std::ios::app);
            if (outputFile_results.is_open())
            {
                outputFile_results << gmem_offset << "," << shmem_offset << "," << total << std::endl;
                outputFile_results.close();
            }
            else
            {
                std::ostringstream oss;
                oss << "Unable to read file " << filename_results << " to write performance number" << std::endl;
                throw std::runtime_error(oss.str());
            }
        }
    }

    // Deallocation
    cudaFree(input_array);
    cudaFree(compute_time);

    return 0;
}
