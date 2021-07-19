/*
 * inversek2j.cpp
 *
 * Created on: Sep. 10 2013
 * Author: Amir Yazdanbakhsh <yazdanbakhsh@wisc.edu>
 */

#include <iostream>
#include <cstdlib>
#include <fstream>
#include <string>
#include <chrono>
#include "kinematics.hpp"


int main(int argc, const char* argv[]) {
    int n;
    std::string inputFilename = argv[1];
    std::string outputFilename = argv[2];

    // Prepare the input file for reading the theta data
    std::ifstream inputFileHandler (inputFilename, std::ifstream::in);

	// Prepare the output file for writting the theta values
    std::ofstream outputFileHandler;
    outputFileHandler.open(outputFilename);

    // First line defines the number of entries
    inputFileHandler >> n;

    float* t1t2xy = (float*)malloc(n * 4 * sizeof(float));

    if (t1t2xy == NULL) {
        std::cerr << "Cannot allocate memory for the coordinates an angles!" << std::endl;
        return -1;
    }

    for (int i = 0; i < n * 4; i += 4) {
        float theta1, theta2;
        inputFileHandler >> theta1 >> theta2;

        t1t2xy[i] = theta1;
        t1t2xy[i + 1] = theta2;

        forwardk2j(t1t2xy[i + 0], t1t2xy[i + 1], t1t2xy + (i + 2), t1t2xy + (i + 3));
    }

    auto begin = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < n * 4; i += 4) {
        inversek2j(t1t2xy[i + 2], t1t2xy[i + 3], t1t2xy + (i + 0), t1t2xy + (i + 1));
    }
    auto end = std::chrono::high_resolution_clock::now();
    std::cout << std::chrono::duration_cast<std::chrono::microseconds>(end-begin).count() << " us" << std::endl;

    for (int i = 0; i < n * 4; i += 4) {
        outputFileHandler <<  t1t2xy[i+0] << "\t" << t1t2xy[i+1] << "\n";
    }

    inputFileHandler.close();
    outputFileHandler.close();

    free(t1t2xy);

    return 0;
}
