#include <stdio.h>
#include <stdint.h>
#include "parfileReader.h"
#include <cuda_pipeline_primitives.h>

// This is a function that checks of results are consistent across kernel implementation.
template <typename T>
void check_result(T *input, T *output_gpu, int n, std::string kernel_version)
{
    size_t err = 0;
    T tol = 1e-4;
    for (int i = 0; i < n; i++)
    {
        T rel_err = abs((input[i] - output_gpu[n - i - 1]) / input[i]);
        if (rel_err > tol)
        {
            printf("Error %s i: %d, input: %e, output_gpu: %e, relative err: %e\n ",
                   kernel_version.c_str(),
                   i,
                   input[i],
                   output_gpu[n - i - 1],
                   rel_err);
            err++;
        }
    }
    if (!err)
        printf("Success %s\n", kernel_version.c_str());
}

// Copy elements from one array in global memory into another array also in global memory.
// No shared memory used.
__global__ void copy_ldg(float *input, float *output, int n)
{
    int idx = threadIdx.x;

    // Write data back to global memory
    output[n - idx - 1] = input[idx];
}

// Copy elements from one array in global into shared memory, then write back to global memory.
template <int stride>
__global__ void copy_shmem(float *input, float *output, int n)
{
    // Declare shared memory (size is known at compile time).
    __shared__ float shared_array[1024];

    // Load data into shared memory.
    int idx = threadIdx.x;
    shared_array[idx * stride] = input[idx];

    // Wait until all threads in the block are done loading their data to shared memory.
    __syncthreads();

    // Write data back to global memory
    output[n - idx - 1] = shared_array[idx * stride];
}

// Copy elements from global to shared, then shared to shared, then back to global memory.
// This kernel illustrates the need to go through a register when copying data between two shared memory arrays. 
template <int stride>
__global__ void copy_shmem2shmem(float *input, float *output, int n)
{
    // Declare two shared memory arrays.
    __shared__ float shared_array0[1024];
    __shared__ float shared_array1[1024];

    int idx = threadIdx.x;
    shared_array0[idx * stride] = input[idx];
    __syncthreads();

    shared_array1[1024 - idx * stride] = shared_array0[idx * stride];
    __syncthreads();

    // Write data back to global memory. 
    output[n - idx - 1] = shared_array1[1024 - idx * stride];
}

// Copy elements from one array in global memory to shared memory using LDGSTS.
// 4-byte strided access. 
template <int stride>
__global__ void copy_ldgsts(float *input, float *output, int n)
{
    // Declare shared memory
    __shared__ float shared_array[1024];

    int idx = threadIdx.x;

    // Launch asynchronous copies
    __pipeline_memcpy_async(shared_array + idx * stride, input + idx, sizeof(float));
    __pipeline_commit();

    // Thread waits until asynchronous copy is completed
    __pipeline_wait_prior(0);

    // Wait until all threads in the block are done loading their data to shared memory
    __syncthreads();

    // Write data back to global memory
    output[n - idx - 1] = shared_array[idx * stride];
}

// Copy batches of elements from one array into another, going through shared memory using LDGSTS
// Loading data with three possible sizes: 4, 8, and 16 bytes. 
// LDGSTS only works with one of these three data sizes
template <typename T, int nbytes_load>
__global__ void copy_ldgsts_bypass_l1(T *input, T *output, int n)
{
    // Declare shared memory
    __shared__ __align__(nbytes_load) T shared_array[1024];
    // __shared__ T shared_array[1024];

    // Make sure the number of bytes loaded per LDGSTS call is a multiple of sizeof(T)
    // For instance, T=int4, nbytes_load must be 16
    static_assert(nbytes_load % sizeof(T) == 0);
    int nelement_load_per_thread = nbytes_load / sizeof(T);
    int tx = threadIdx.x;
    int tx_load = nelement_load_per_thread * threadIdx.x; // Assumes n % n_load_per_thread = 0

    if (tx_load < n)
    {
        // Launch asynchronous copies
        __pipeline_memcpy_async(shared_array + tx_load, input + tx_load, nbytes_load);
    }
    __pipeline_commit();

    // Thread waits until asynchronous copy is completed
    __pipeline_wait_prior(0);

    // Wait until all threads in the block are done loading their data to shared memory
    __syncthreads();

    // Write data back to global memory
    output[n - tx - 1] = shared_array[tx];
}

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

    // Sizes
    const int warp_size = 32; // 32 threads in a warp
    const int nwarp = 1;
    const int n_element = warp_size * nwarp;

    // Block / Grid
    int block = n_element;
    int grid = (n_element + block - 1) / block;
    printf("ntotal: %d, block: %d, grid: %d\n", n_element, block, grid);

    // Kernel to profile
    std::string kernel_type = par.getString("kernel_type");

    // Allocation
    float *input, *output_gpu;
    cudaMallocManaged((void **)&input, n_element * sizeof(float));
    cudaMallocManaged((void **)&output_gpu, n_element * sizeof(float));

    // Initialize input
    for (int i = 0; i < n_element; i++)
        input[i] = 1.0f * i;

    // LDG
    memset(output_gpu, 0, n_element);
    copy_ldg<<<grid, block>>>(input, output_gpu, n_element);
    cudaDeviceSynchronize();
    check_result<float>(input, output_gpu, n_element, "copy_shmem");

    // LDG + STS
    // STS are done with a stride, this is done to illustrate how to handle shared memory bank conflicts
    if (kernel_type == "ldg_sts")
    {
        // Shmem stride = 1
        memset(output_gpu, 0, n_element);
        copy_shmem<1><<<grid, block>>>(input, output_gpu, n_element);
        cudaDeviceSynchronize();
        check_result<float>(input, output_gpu, n_element, "copy_shmem_stride1");

        // Shmem stride = 2
        memset(output_gpu, 0, n_element);
        copy_shmem<2><<<grid, block>>>(input, output_gpu, n_element);
        cudaDeviceSynchronize();
        check_result<float>(input, output_gpu, n_element, "copy_shmem_stride2");

        // Shmem stride = 4
        memset(output_gpu, 0, n_element);
        copy_shmem<4><<<grid, block>>>(input, output_gpu, n_element);
        cudaDeviceSynchronize();
        check_result<float>(input, output_gpu, n_element, "copy_shmem_stride4");

        // Shmem stride = 8
        memset(output_gpu, 0, n_element);
        copy_shmem<8><<<grid, block>>>(input, output_gpu, n_element);
        cudaDeviceSynchronize();
        check_result<float>(input, output_gpu, n_element, "copy_shmem_stride8");

        // Shmem2shmem
        memset(output_gpu, 0, n_element);
        copy_shmem2shmem<1><<<grid, block>>>(input, output_gpu, n_element);
        cudaDeviceSynchronize();
        check_result<float>(input, output_gpu, n_element, "copy_shmem2shmem");
    }

    // LDGSTS
    else if (kernel_type == "ldgsts")
    {
        memset(output_gpu, 0, n_element);
        copy_ldgsts<1><<<grid, block>>>(input, output_gpu, n_element);
        cudaDeviceSynchronize();
        check_result<float>(input, output_gpu, n_element, "copy_ldgsts");

        // LDGSTS "float" (4 bytes)
        memset(output_gpu, 0, n_element);
        copy_ldgsts_bypass_l1<float, 4><<<grid, block>>>(input, output_gpu, n_element);
        cudaDeviceSynchronize();
        check_result<float>(input, output_gpu, n_element, "copy_ldgsts_bypass_l1 (4 bytes)");

        // LDGSTS "float2" (8 bytes)
        memset(output_gpu, 0, n_element);
        copy_ldgsts_bypass_l1<float, 8><<<grid, block>>>(input, output_gpu, n_element);
        cudaDeviceSynchronize();
        check_result<float>(input, output_gpu, n_element, "copy_ldgsts_bypass_l1 (8 bytes)");

        // LDGSTS "float4" (16 bytes)
        memset(output_gpu, 0, n_element);
        copy_ldgsts_bypass_l1<float, 16><<<grid, block>>>(input, output_gpu, n_element);
        cudaDeviceSynchronize();
        check_result<float>(input, output_gpu, n_element, "copy_ldgsts_bypass_l1 (16 bytes)");
    }
    else 
    {
        throw std::runtime_error("No kernel to run, please select 'ldg_sts' or 'ldgsts' for 'kernel_type'"\n);        
    }

    // Deallocation
    cudaFree(input);
    cudaFree(output_gpu);

    return 0;
}
