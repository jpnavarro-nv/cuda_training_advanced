#include <stdio.h>
#include <stdint.h>
#include "parfileReader.h"

// Copy batches of elements from one array into another, for loop unrolled
template <int batch, int nwarp>
__global__ void ldg_copy_batch_unrolled(int *input, int *output)
{
#pragma unroll
    for (int i = 0; i < batch; i++)
        output[32 * nwarp * i + threadIdx.x] = input[32 * nwarp * i + threadIdx.x];
}

// Copy batches of elements from one array into another, for loop unrolled
template <int batch, int nwarp>
__global__ void ldg_copy_batch_restrict(int *__restrict__ input, int *__restrict__ output)
{
#pragma unroll
    for (int i = 0; i < batch; i++)
        output[32 * nwarp * i + threadIdx.x] = input[32 * nwarp * i + threadIdx.x];
}

// Copy batches of elements from one array into another, no unroll
__global__ void ldg_copy_batch_no_unroll(int *input, int *output, int batch, int nwarp)
{
    // 'batch' is not known at compile time, the compiler cannot unroll the loop inside the kernel
    for (int i = 0; i < batch; i++)
        output[32 * nwarp * i + threadIdx.x] = input[32 * nwarp * i + threadIdx.x];
}

// Copy batches of elements from one array into another using int4 (16-bytes loads/stores)
__global__ void ldg_copy_batch_int4(int *input, int *output)
{
    int4 *input4 = (int4 *)input;
    int4 *output4 = (int4 *)output;
    output4[threadIdx.x] = input4[threadIdx.x];
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
    const int warp_size = 32;                              // 32 threads in a warp
    const int batch_size = 4;                              // 16-byte access (4 elements of type 'int')
    const int n_element_per_warp = warp_size * batch_size; // Each thread copies 4 elements, and so each warp copies 128 elements
    const int n_element = n_element_per_warp * NWARP;      // Total number of elements to copy from input to output. NWARP is defined at compile time
    int n_in = n_element;
    int n_out = n_element;

    // Block / Grid
    int block = std::min(1024, warp_size * NWARP); // NWARP is a compile-time defined variable (needed is used for template)
    int grid = (n_element + batch_size * block - 1) / (batch_size * block);
    printf("ntotal: %d, block: %d, grid: %d\n", n_element, block, grid);

    // Allocation
    int *input, *output, *output_unroll, *output_restrict, *output_int4;
    cudaMallocManaged((void **)&input, n_in * sizeof(int));
    cudaMallocManaged((void **)&output, n_out * sizeof(int));
    cudaMallocManaged((void **)&output_unroll, n_out * sizeof(int));
    cudaMallocManaged((void **)&output_restrict, n_out * sizeof(int));
    cudaMallocManaged((void **)&output_int4, n_out * sizeof(int));

    // Initialize input
    for (int i = 0; i < n_in; i++)
        input[i] = i;

    // Initialize outputs
    memset(output_unroll, 0, n_element);
    memset(output_restrict, 0, n_element);
    memset(output, 0, n_element);
    memset(output_int4, 0, n_element);

    // Launch batch copy kernels with unrolled loop
    ldg_copy_batch_unrolled<batch_size, NWARP><<<grid, block>>>(input, output_unroll);

    // Launch batch copy kernels with unrolled loop and restrict keyword
    ldg_copy_batch_restrict<batch_size, NWARP><<<grid, block>>>(input, output_restrict);

    // Launch non-templated batch copy kernel
    int batch = par.getInt("batch", 4);
    ldg_copy_batch_no_unroll<<<grid, block>>>(input, output, batch, NWARP);

    // Launch int4 copy kernel
    ldg_copy_batch_int4<<<grid, block>>>(input, output_int4);

    // Block CPU until all kernels from default stream are executed
    cudaDeviceSynchronize();

    // Check results
    int err = 0;
    for (int i = 0; i < n_element; i++)
    {
        if (output_unroll[i] != output_int4[i] || output_restrict[i] != output_int4[i] || output[i] != output_int4[i])
        {
            printf("Error i: %d, output_unroll: %d, output_restrict: %d, output: %d, output_int4: %d\n", i, output_unroll[i], output_restrict[i], output[i], output_int4[i]);
            err++;
        }
    }
    if (!err)
        printf("Success\n");

    // Deallocation
    cudaFree(input);
    cudaFree(output);
    cudaFree(output_unroll);
    cudaFree(output_restrict);
    cudaFree(output_int4);

    return 0;
}
