#include <cuda_pipeline_primitives.h>
#include <stdio.h>
#include <stdint.h>
#include "parfileReader.h"

// Stencil radius for derivative
#define RADIUS 1

// Dummy kernel that only copies elements from one array into another
template <int radius>
__global__ void ldg_deriv_copy(float *input, float *output, int n0, int n1, int n0_pad)
{
    int tx = threadIdx.x + radius;
    int ty = threadIdx.y;
    if (tx < radius + n0 && ty < n1)
    {
        int tid = ty * n0_pad + tx;
        output[tid] = input[tid];
    }
}

// Finite-difference kernel that computes a first-order derivative in the leading dimension
template <int radius>
__global__ void ldg_deriv0(float *input, float *output, int n0, int n1, int n0_pad)
{
    int tx = threadIdx.x + radius;
    int ty = threadIdx.y;
    if (tx < radius + n0 && ty < n1)
    {
        int tid = ty * n0_pad + tx;
        output[tid] = input[tid] - input[tid - 1];
    }
}

// Finite-difference kernel that computes a first-order derivative in the slow dimension
template <int radius>
__global__ void ldg_deriv1(float *input, float *output, int n0, int n1, int n0_pad)
{
    int tx = threadIdx.x + radius;
    int ty = threadIdx.y;
    if (tx < radius + n0 && ty > 0 && ty < n1)
    {
        int tid = ty * n0_pad + tx;
        output[tid] = input[tid] - input[tid - n0_pad];
    }
}

int main(int argc, char **argv)
{
    // Read parameters from command line
    parfileReader par(argc, argv);

    // Number of elements in float format in one cache line
    const int cache_line_size_float = 128 / sizeof(float);

    // Sizes
    int n0 = par.getInt("n0"); // Number of points in the leading dimension
    int n1 = par.getInt("n1");
    int n0_pad = n0 + 2 * RADIUS; // Pad by adding half-stencil number of points on each side of the leading dimension
    int n_total = n0_pad * n1;
    int n0_pad_align = (n0 + cache_line_size_float - 1) / cache_line_size_float * cache_line_size_float; // Put additional padding at the end of the leading dimension for cache line alignment
    int n_total_align = n0_pad_align * n1;

    // Compute lead pad:
    // Add padding at the beginning of an array to ensure that the first point that needed to be updated aligns with the beginning of a cache line
    int lead_pad = RADIUS % cache_line_size_float;
    lead_pad == 0 ? lead_pad = 0 : lead_pad = cache_line_size_float - lead_pad;

    // Block / Grid
    const int block_dim0 = 32;
    const int block_dim1 = 32;
    dim3 block(block_dim0, block_dim1);
    dim3 grid(
        (n0 + block_dim0 - 1) / block_dim0,
        (n1 + block_dim1 - 1) / block_dim1);
    printf("blockx: %d, blocky: %d, gridx: %d, gridy: %d, radius: %d, lead_pad: %d\n", block.x, block.y, grid.x, grid.y, RADIUS, lead_pad);
    printf("n0: %d, n1: %d, n0_pad: %d, n0_pad_align: %d\n", n0, n1, n0_pad, n0_pad_align);

    // Allocation
    float *input, *output, *input_lp, *output_lp, *input_align, *output_align;
    cudaMallocManaged((void **)&input, n_total * sizeof(float));                           // No alignment
    cudaMallocManaged((void **)&output, n_total * sizeof(float));                          // No alignement
    cudaMallocManaged((void **)&input_lp, (n_total + lead_pad) * sizeof(float));           // Lead padding only
    cudaMallocManaged((void **)&output_lp, (n_total + lead_pad) * sizeof(float));          // Lead padding only
    cudaMallocManaged((void **)&input_align, (n_total_align + lead_pad) * sizeof(float));  // Lead padding + padding in the leading dimension
    cudaMallocManaged((void **)&output_align, (n_total_align + lead_pad) * sizeof(float)); // Lead padding + padding in the leading dimension

    // Shift pointers by lead pad
    input_lp += lead_pad;
    output_lp += lead_pad;
    input_align += lead_pad;
    output_align += lead_pad;

    // Array initialization
    memset(input, 0, n_total);
    memset(input_lp, 0, n_total);
    memset(input_align, 0, n_total_align);
    memset(output, 0, n_total);
    memset(output_lp, 0, n_total);
    memset(output_align, 0, n_total_align);

    // Launch kernel that only performs copies from input -> output
    if (par.getString("kernel") == "copy")
    {
        ldg_deriv_copy<RADIUS><<<grid, block>>>(input, output, n0, n1, n0_pad);
        ldg_deriv_copy<RADIUS><<<grid, block>>>(input_lp, output_lp, n0, n1, n0_pad);
        ldg_deriv_copy<RADIUS><<<grid, block>>>(input_align, output_align, n0, n1, n0_pad_align);
    }
    // Launch kernel that computes first-order derivative in the leading dimension on input array 
    else if (par.getString("kernel") == "deriv0")
    {
        ldg_deriv0<RADIUS><<<grid, block>>>(input, output, n0, n1, n0_pad);
        ldg_deriv0<RADIUS><<<grid, block>>>(input_lp, output_lp, n0, n1, n0_pad);
        ldg_deriv0<RADIUS><<<grid, block>>>(input_align, output_align, n0, n1, n0_pad_align);
    }
    else if (par.getString("kernel") == "deriv1")
    {
        /* 
        TO DO: 
        - Implement a finite-difference derivative operator in the slowest dimension (n1)
        Note: Be careful about the bounds (we did not pad by RADIUS in the slow dimension). At which index should the first and last points in n1 direction be computed? 
        - Vary the number of points in the 0- and 1-dimension and observe the effect on the L1 hit rate
        */
        ldg_deriv1<RADIUS><<<grid, block>>>(input, output, n0, n1, n0_pad);
        ldg_deriv1<RADIUS><<<grid, block>>>(input_lp, output_lp, n0, n1, n0_pad);
        ldg_deriv1<RADIUS><<<grid, block>>>(input_align, output_align, n0, n1, n0_pad_align);
    }
    else
    {
        throw std::runtime_error("Please set kernel tag among 'copy', 'deriv0' or 'deriv1'\n");        
    }

    // Force CPU to wait for kernels launched on default stream
    cudaDeviceSynchronize();

    // Check results
    int err = 0;
    for (int i1 = 0; i1 < n1; i1++)
        for (int i0 = 0; i0 < n0; i0++)
        {
            int idx = i1 * n0_pad + i0 + RADIUS;
            int idx_align = i1 * n0_pad_align + i0 + RADIUS;
            if (output[idx] != output_lp[idx] || output[idx] != output_align[idx_align])
            {
                printf("Error i0: %d, i1: %d, output: %f, output_lp: %f, output_align: %f\n", i0, i1, output[idx], output_lp[idx], output_align[idx_align]);
                err++;
            }
        }
    if (!err)
        printf("Success\n");

    // Shift array pointers back by lead_pad
    input_lp -= lead_pad;
    output_lp -= lead_pad;
    input_align -= lead_pad;
    output_align -= lead_pad;

    // Deallocate
    cudaFree(input_lp);
    cudaFree(output_lp);
    cudaFree(input_align);
    cudaFree(output_align);

    return 0;
}