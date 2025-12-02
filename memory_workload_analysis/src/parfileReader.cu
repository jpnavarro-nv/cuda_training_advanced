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
#include "parfileReader.h"
#include <iomanip>

// Constructor
parfileReader::parfileReader(int argc, char *argv[])
{
    // Position in argument lines after reading parfile
    int shift = 0;

    // Read name of parfile
    for (int i = 1; i < argc; i++)
    {
        std::string first_four = std::string(argv[i]).substr(0, 4);
        // printf("first_four: %s\n", first_four.c_str());
        if (first_four == "par=")
        {
            // _parfileName = std::string(argv[i]);
            _parfileName = std::string(argv[i]).substr(4);
            shift += 1;
        }
    }

    // Check that the parfile is not empty
    if (_parfileName.empty())
    {
        // printf("[parfileReader] Warning: parfile '%s' provided by user is empty\n", _parfileName.c_str());
    }

    // Open textfile
    std::ifstream fd(_parfileName, std::fstream::in);
    if (!fd.is_open())
    {
        // printf("[parfileReader] Warning: could not open parfile '%s'\n", _parfileName.c_str());
    }

    std::string current_line;
    while (std::getline(fd, current_line))
    {
        if (std::isalpha(current_line.c_str()[0]))
        {
            _fileInput.push_back(current_line);
        }
    }

    // Close text file
    fd.close();

    // Read any other argument after the parfile
    for (int i = shift + 1; i < argc; i += 1)
    {
        current_line = std::string(argv[i]);
        _fileInput.push_back(current_line);
    }
}

// Helper function for getInt
bool parfileReader::findInt(std::string tagName, int &val_out)
{
    // Initialize value to false
    bool found_tag = false;

    // Loop over textfile lines
    for (int iline = 0; iline < _fileInput.size(); iline++)
    {
        // Check if we found an "=" sign on this line
        std::string current_line = _fileInput[iline];
        std::string eq_sign = std::string("=");
        std::size_t found = current_line.find(eq_sign);

        // Add a space after equal sign
        current_line.insert(found + 1, " ");

        // Parse current line
        if (strncmp(current_line.c_str(), tagName.c_str(), strlen(tagName.c_str())) == 0) // Add condition that next character should be a space or a "="
        {
            // Check if next character is a space or a equal sign
            char next_str = current_line[strlen(tagName.c_str())];
            if (std::isspace(next_str) || next_str == '=')
            {
                // Create istringstream
                std::istringstream inputStringStream;
                inputStringStream.str(current_line);
                std::string val_str;

                // Read until end of line, write string values in val_str
                while (!inputStringStream.eof())
                    inputStringStream >> val_str;

                // Convert value to integer
                val_out = stoi(val_str);
                found_tag = true;
            }
        }
    }
    return found_tag;
}

// Single integer value with default value
int parfileReader::getInt(std::string tagName, int val_default)
{
    // Initialize value
    int val_out;
    bool found_tag = findInt(tagName, val_out);

    // Check if a tag value was found
    if (found_tag)
        return val_out;
    else
    {
        // Use default value
        return val_default;
    }
}

// Single integer value
int parfileReader::getInt(std::string tagName)
{
    // Initialize value
    int val_out;
    bool found_tag = findInt(tagName, val_out);

    if (!found_tag)
    {
        printf("[parfileReader::getInt] Could not find a value for tag '%s'\n", tagName.c_str());
        throw std::runtime_error(" ");
    }
    else
        return val_out;
}

// Get list of ints
std::vector<int> parfileReader::getInts(std::string tagName)
{
    // Initialize value
    std::vector<int> val_out;
    bool found_tag = false;
    std::string str_tmp;

    // Loop over textfile lines
    for (int iline = 0; iline < _fileInput.size(); iline++)
    {
        // Check if we found an "=" sign on this line
        std::string current_line = _fileInput[iline];
        std::string eq_sign = std::string("=");
        std::size_t found = current_line.find(eq_sign);

        // Add a space after equal sign
        current_line.insert(found + 1, " ");

        // Parse current line
        if (strncmp(current_line.c_str(), tagName.c_str(), strlen(tagName.c_str())) == 0)
        {
            // Check if next character is a space or a equal sign
            char next_str = current_line[strlen(tagName.c_str())];
            if (std::isspace(next_str) || next_str == '=')
            {
                // Override previous definitions
                val_out.clear();
                int len = current_line.length();                                   // Get the number of characters on this line
                int wcount = count(current_line.begin(), current_line.end(), ' '); // Count the number of white spaces
                remove(current_line.begin(), current_line.end(), ' ');             // Remove white spaces
                current_line.resize(len - wcount);                                 // Update the number of elements for this line
                remove(current_line.begin(), current_line.end(), ' ');

                // Create stringstream object
                std::stringstream inputStringStream(current_line);

                // Shift pointer to be after the "=" sign
                size_t offset_line = strlen(tagName.c_str()) + 1;
                inputStringStream.seekg(offset_line);

                while (getline(inputStringStream, str_tmp, ','))
                {
                    val_out.push_back(stoi(str_tmp));
                }

                found_tag = true; // List of values was found
            }
        }
    }
    if (!found_tag)
    {
        printf("[parfileReader::getInts] Could not find values for tag '%s'\n", tagName.c_str());
        throw std::runtime_error(" ");
    }
    else
        return val_out;
}

// Helper function for getFloat
bool parfileReader::findFloat(std::string tagName, float &val_out)
{
    // Initialize value to false
    bool found_tag = false;

    // Loop over textfile lines
    for (int iline = 0; iline < _fileInput.size(); iline++)
    {
        // Check if we found an "=" sign on this line
        std::string current_line = _fileInput[iline];
        std::string eq_sign = std::string("=");
        std::size_t found = current_line.find(eq_sign);

        // Add a space after equal sign
        current_line.insert(found + 1, " ");

        // Parse current line
        if (strncmp(current_line.c_str(), tagName.c_str(), strlen(tagName.c_str())) == 0) // Add condition that next character should be a space or a "="
        {
            // Check if next character is a space or a equal sign
            char next_str = current_line[strlen(tagName.c_str())];
            if (std::isspace(next_str) || next_str == '=')
            {
                // Create istringstream
                std::istringstream inputStringStream;
                inputStringStream.str(current_line);
                std::string val_str;

                // Read until end of line, write string values in val_str
                while (!inputStringStream.eof())
                    inputStringStream >> val_str;

                // Convert value to integer
                val_out = stof(val_str);
                found_tag = true;
            }
        }
    }
    return found_tag;
}

// Get list single float value
float parfileReader::getFloat(std::string tagName)
{
    // Initialize value
    float val_out;
    bool found_tag = findFloat(tagName, val_out);

    // Check if value was found
    if (!found_tag)
    {
        printf("[parfileReader::getFloat] Could not find values for tag '%s'\n", tagName.c_str());
        throw std::runtime_error(" ");
    }
    else
        return val_out;
}

// Get list single float value
float parfileReader::getFloat(std::string tagName, float val_default)
{
    // Initialize value
    float val_out;
    bool found_tag = findFloat(tagName, val_out);

    if (found_tag)
        return val_out;
    else
    {
        // printf("[parfileReader::getFloat] Using default value of %f for tag %s\n", val_default, tagName.c_str());
        return val_default;
    }
}

// Get list of floats
std::vector<float> parfileReader::getFloats(std::string tagName)
{
    // Initialize value
    std::vector<float> val_out;
    bool found_tag = false;
    std::string str_tmp;

    // Loop over textfile lines
    for (int iline = 0; iline < _fileInput.size(); iline++)
    {
        // Check if we found an "=" sign on this line
        std::string current_line = _fileInput[iline];
        std::string eq_sign = std::string("=");
        std::size_t found = current_line.find(eq_sign);

        // Add a space after equal sign
        current_line.insert(found + 1, " ");

        // Parse current line
        if (strncmp(current_line.c_str(), tagName.c_str(), strlen(tagName.c_str())) == 0)
        {
            // Check if next character is a space or a equal sign
            char next_str = current_line[strlen(tagName.c_str())];
            if (std::isspace(next_str) || next_str == '=')
            {
                // Override previous definitions
                val_out.clear();
                int len = current_line.length();                                   // Get the number of characters on this line
                int wcount = count(current_line.begin(), current_line.end(), ' '); // Count the number of white spaces
                remove(current_line.begin(), current_line.end(), ' ');             // Remove white spaces
                current_line.resize(len - wcount);                                 // Update the number of elements for this line
                remove(current_line.begin(), current_line.end(), ' ');

                // Create stringstream object
                std::stringstream inputStringStream(current_line);

                // Shift pointer to be after the "=" sign
                size_t offset_line = strlen(tagName.c_str()) + 1;
                inputStringStream.seekg(offset_line);

                while (getline(inputStringStream, str_tmp, ','))
                    val_out.push_back(stof(str_tmp));

                // Update tag value
                found_tag = true;
            }
        }
    }
    if (!found_tag)
    {
        printf("[parfileReader::getFloats] Could not find values for tag '%s'\n", tagName.c_str());
        throw std::runtime_error(" ");
    }
    else
        return val_out;
}

// Helper function for getInt
bool parfileReader::findString(std::string tagName, std::string &val_out)
{
    // Initialize value to false
    bool found_tag = false;

    // Loop over textfile lines
    for (int iline = 0; iline < _fileInput.size(); iline++)
    {
        // Check if we found an "=" sign on this line
        std::string current_line = _fileInput[iline];
        std::string eq_sign = std::string("=");
        std::size_t found = current_line.find(eq_sign);

        // Add a space after equal sign
        current_line.insert(found + 1, " ");

        // Parse current line
        if (strncmp(current_line.c_str(), tagName.c_str(), strlen(tagName.c_str())) == 0) // Add condition that next character should be a space or a "="
        {
            // Check if next character is a space or a equal sign
            char next_str = current_line[strlen(tagName.c_str())];
            if (std::isspace(next_str) || next_str == '=')
            {
                // Create istringstream
                std::istringstream inputStringStream;
                inputStringStream.str(current_line);
                std::string val_str;

                // Read until end of line, write string values in val_str
                while (!inputStringStream.eof())
                    inputStringStream >> val_str;

                // Convert value to integer
                val_out = val_str;
                found_tag = true;
            }
        }
    }
    return found_tag;
}

// Get string value
std::string parfileReader::getString(std::string tagName)
{
    // Initialize value
    std::string val_out;
    bool found_tag = findString(tagName, val_out);

    if (!found_tag)
    {
        printf("[parfileReader::getString] Could not find a value for tag '%s'\n", tagName.c_str());
        throw std::runtime_error(" ");
    }
    else
        return val_out;
}

// Get string value
std::string parfileReader::getString(std::string tagName, std::string val_default)
{
    // Initialize value
    std::string val_out;
    bool found_tag = findString(tagName, val_out);

    // Check if a tag value was found
    if (found_tag)
        return val_out;
    else
        // Use default value
        return val_default;
}