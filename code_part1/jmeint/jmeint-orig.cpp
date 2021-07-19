/*
 * jmeint.cpp
 *
 * Created on: Sep 9, 2013
 * Author: Amir Yazdanbakhsh <a.yazdanbakhsh@gatech.edu>
 */

#include <fstream>
#include <iostream>
#include <chrono>
#include "tritri.hpp"

int main(int argc, char* argv[]) {
    std::string inputFilename = argv[1];
    std::string outputFilename = argv[2];

    // Prepare the input file for reading the input data
    std::ifstream inputFileHandler(inputFilename, std::ifstream::in);

    // Prepare output file for writting the result values
    std::ofstream outputFileHandler;
    outputFileHandler.open(outputFilename);
    outputFileHandler.precision(5);

    // First line defines the number of enteries
    int n;
    inputFileHandler >> n;

    // Allocate memory to store the data
    float* xyz = (float*)malloc(n * 18 * sizeof(float));
    if(xyz == NULL) {
        std::cout << "Error allocating memory." << std::endl;
        return -1 ;
    }

    // Read input data
    for (int i = 0; i < n; i++) {
        float a[18];
        inputFileHandler >> a[0]  >> a[1]  >> a[2]  >> a[3]  >> a[4]  >> a[5]
                         >> a[6]  >> a[7]  >> a[8]  >> a[9]  >> a[10] >> a[11]
                         >> a[12] >> a[13] >> a[14] >> a[15] >> a[16] >> a[17];
        for (int j = 0; j < 18; j++) {
            xyz[i * 18 + j] = a[j];
        }
    }

    short int res[n] = {};

    auto begin = std::chrono::high_resolution_clock::now();
    for (int i = 0 ; i < (n * 6 * 3); i += 6 * 3) {
        res[i/18] = tri_tri_intersect(
                        xyz + i + 0 * 3, xyz + i + 1 * 3, xyz + i + 2 * 3,
                        xyz + i + 3 * 3, xyz + i + 4 * 3, xyz + i + 5 * 3);
    }
    auto end = std::chrono::high_resolution_clock::now();
    std::cout << std::chrono::duration_cast<std::chrono::microseconds>(end-begin).count() << " us" << std::endl;

    // Write output data
    for (int i = 0; i < n; i++) {
        outputFileHandler << res[i] << std::endl;
    }

    inputFileHandler.close();
    outputFileHandler.close();

    free(xyz);

    return 0;
}
