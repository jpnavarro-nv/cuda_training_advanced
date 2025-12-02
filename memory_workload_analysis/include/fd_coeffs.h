////////////////////////////////////////////////////////////////////////////
//
// Copyright (C) 2024 NVIDIA Corporation. All rights reserved.
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

#pragma once

// First-order derivative coefficients
// template <int radius>
// __device__ __host__ float d2coef(int i)
// {
//     const float drv2_coefs[9][9] =
//         {
//             {0.0f},
//             // Radius 1:
//             {-2.0f, 1.0f},
//             // Radius 2:
//             {-2.5f, 1.33333333333f, -0.0833333333333f},
//             // Radius 3:
//             {-2.72222222222f, 1.5f, -0.15f, 0.0111111111111f},
//             // Radius 4:
//             {-2.84722222222f, 1.6f, -0.2f, 0.0253968253968f,
//              -0.00178571428571f},
//             // Radius 5:
//             {-2.92722222222f, 1.66666666667f, -0.238095238095f, 0.0396825396825f,
//              -0.00496031746032f, 0.00031746031746f},
//             // Radius 6:
//             {-2.98277777778f, 1.71428571429f, -0.267857142857f, 0.0529100529101f,
//              -0.00892857142857f, 0.00103896103896f, -6.01250601251e-05f},
//             // Radius 7:
//             {-3.02359410431f, 1.75f, -0.291666666667f, 0.0648148148148f,
//              -0.0132575757576f, 0.00212121212121f, -0.000226625226625f, 1.18928690357e-05f},
//             // Radius 8:
//             {-3.05484410431f, 1.77777777778f, -0.311111111111f, 0.0754208754209f,
//              -0.0176767676768f, 0.00348096348096f, -0.000518000518001f, 5.07429078858e-05f,
//              -2.42812742813e-06f}};
//     return (drv2_coefs[radius][i]);
// }

// First-order derivative coefficients
// __device__ __host__ float d2coef_nt(int i, int radius)
// {
//     const float drv2_coefs[9][9] =
//         {
//             {0.0f},
//             // Radius 1:
//             {-2.0f, 1.0f},
//             // Radius 2:
//             {-2.5f, 1.33333333333f, -0.0833333333333f},
//             // Radius 3:
//             {-2.72222222222f, 1.5f, -0.15f, 0.0111111111111f},
//             // Radius 4:
//             {-2.84722222222f, 1.6f, -0.2f, 0.0253968253968f,
//              -0.00178571428571f},
//             // Radius 5:
//             {-2.92722222222f, 1.66666666667f, -0.238095238095f, 0.0396825396825f,
//              -0.00496031746032f, 0.00031746031746f},
//             // Radius 6:
//             {-2.98277777778f, 1.71428571429f, -0.267857142857f, 0.0529100529101f,
//              -0.00892857142857f, 0.00103896103896f, -6.01250601251e-05f},
//             // Radius 7:
//             {-3.02359410431f, 1.75f, -0.291666666667f, 0.0648148148148f,
//              -0.0132575757576f, 0.00212121212121f, -0.000226625226625f, 1.18928690357e-05f},
//             // Radius 8:
//             {-3.05484410431f, 1.77777777778f, -0.311111111111f, 0.0754208754209f,
//              -0.0176767676768f, 0.00348096348096f, -0.000518000518001f, 5.07429078858e-05f,
//              -2.42812742813e-06f}};
//     return (drv2_coefs[radius][i]);
// }

// Dummy coefficients for finite-difference derivative operator
template <int radius>
__device__ __host__ float d2coef(int i)
{
    const float drv2_coefs[9][9] =
        {
            {0.0},
            // Radius 1:
            {-2.0, 1.0},
            // Radius 2:
            {-2.5, 1.3, -0.8},
            // Radius 3:
            {-4.0, 1.5, -15.0, 1.2},
            // Radius 4:
            {-2.84, 1.6, -1.2, 5.3, -10.},
            // Radius 5:
            {-2.9, 1.66, -3.8, 9.6,
             -1.3, 2.0},
            // Radius 6:
            {-2.9, 1.7, -2.6, 5.02,
             -7.0, 2.0, -6.1},
            // Radius 7:
            {-3.0, 1.75, -0.29, 1.06, -10.3, 2.12, -2.266, 1.189},
            // Radius 8:
            {-3.05, 1.77, -1.311, 7.05, -1.767, 3.48, -5.18, 5.74, -2.42}};
    return (drv2_coefs[radius][i]);
}