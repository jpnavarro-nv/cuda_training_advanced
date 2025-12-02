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
#include <string>
#include <iostream>
#include <fstream>
#include <vector>
#include <typeinfo>
#include <sstream>
#include <stdio.h>
#include <algorithm>

class parfileReader
{
private:
    std::string _parfileName;
    std::vector<std::string> _fileInput;

public:
    // Constructor
    parfileReader(int argc, char *argv[]);
    ~parfileReader(){};

    // Read integer values from parfile/command line
    int getInt(std::string tagName);
    int getInt(std::string tagName, int val_default);
    std::vector<int> getInts(std::string tag);
    bool findInt(std::string tagName, int &val_out);

    // Read float values from parfile/command line
    float getFloat(std::string tagName);
    float getFloat(std::string tagName, float val_default);
    std::vector<float> getFloats(std::string tagName);
    bool findFloat(std::string tagName, float &val_out);

    // Read strings from parfile/command line
    std::string getString(std::string tagName);
    std::string getString(std::string tagName, std::string val_default);
    bool findString(std::string tagName, std::string &val_out);
};
