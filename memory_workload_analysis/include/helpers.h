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

#include <stdio.h>
#include <cuda.h>
#include <curand.h>

// Generate random data in device memory
template <typename T>
struct Randomizer
{
    curandGenerator_t gen;
    Randomizer(unsigned long long seed = 1234UL)
    {
        curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
        curandSetPseudoRandomGeneratorSeed(gen, seed);
    }
    ~Randomizer()
    {
        curandDestroyGenerator(gen);
    }
    void randomize(T *data, int nelem)
    {
        if (std::is_same_v<T, float>)
        {
            curandSetPseudoRandomGeneratorSeed(gen, 123);
            curandGenerateUniform(gen, (float *)data, nelem);
        }
        else
        {
            curandSetPseudoRandomGeneratorSeed(gen, 123);
            curandGenerateUniformDouble(gen, (double *)data, nelem);
        }
    }
};