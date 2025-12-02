#include <stdio.h>
#include <stdint.h>
#include "parfileReader.h"

// Copy 32 elements (one per thread) from one array into another with a strided access
template <int stride>
__global__ void ldg_copy_stride(int *input, int *output)
{
    output[threadIdx.x] = input[stride * threadIdx.x];
}

int main(int argc, char **argv)
{
    // Parse command line
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
    const int n_element_per_warp = 32;
    const int n_warp = par.getInt("nwarp", 1);
    const int n_element = n_element_per_warp * n_warp;
    const int max_stride = 32; // 32 4-byte accesses
    int n_in = max_stride * n_element;
    int n_out = n_element;

    int block = std::min(1024, n_element);
    int grid = (n_element + block - 1) / block;

    // Allocation
    int *input, *output;
    cudaMallocManaged((void **)&input, n_in * sizeof(int));
    cudaMallocManaged((void **)&output, n_out * sizeof(int));

    // Initialization
    for (int i = 0; i < n_in; i++)
        input[i] = i;

    // Set granularity at which the L2 cache fetches data from global memory
    int l2_granularity = par.getInt("l2", 0);
    if (l2_granularity)
    {
        printf("Setting max granularity: %d\n", l2_granularity);
        cudaDeviceSetLimit(cudaLimitMaxL2FetchGranularity, (size_t)l2_granularity);
    }

    // Launch stride kernels
    ldg_copy_stride<1><<<grid, block>>>(input, output);
    ldg_copy_stride<2><<<grid, block>>>(input, output);
    ldg_copy_stride<3><<<grid, block>>>(input, output);
    ldg_copy_stride<4><<<grid, block>>>(input, output);
    ldg_copy_stride<5><<<grid, block>>>(input, output);
    ldg_copy_stride<6><<<grid, block>>>(input, output);
    ldg_copy_stride<7><<<grid, block>>>(input, output);
    ldg_copy_stride<8><<<grid, block>>>(input, output);
    ldg_copy_stride<9><<<grid, block>>>(input, output);
    ldg_copy_stride<16><<<grid, block>>>(input, output);
    ldg_copy_stride<32><<<grid, block>>>(input, output);
    ldg_copy_stride<64><<<grid, block>>>(input, output);
    ldg_copy_stride<128><<<grid, block>>>(input, output);

    cudaDeviceSynchronize();

    return 0;
}
