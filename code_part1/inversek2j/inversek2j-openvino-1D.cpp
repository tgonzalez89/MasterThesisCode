#include <iostream>
#include <cstdlib>
#include <fstream>
#include <string>
#include <chrono>
#include <cmath>
#include <inference_engine.hpp>
#include "kinematics.hpp"


int main(int argc, const char* argv[]) {
    // Create IE core object and read the network
    InferenceEngine::Core core;
    InferenceEngine::CNNNetwork network = core.ReadNetwork("nn_inversek2j-1D.xml", "nn_inversek2j-1D.bin");
    // Set up the network input
    InferenceEngine::InputsDataMap input_info(network.getInputsInfo());
    auto input_data = input_info.begin()->second;
    input_data->setPrecision(InferenceEngine::Precision::FP32);
    std::string input_name = input_info.begin()->first;
    // Set up the network output
    InferenceEngine::OutputsDataMap output_info(network.getOutputsInfo());
    auto output_data = output_info.begin()->second;
    output_data->setPrecision(InferenceEngine::Precision::FP32);
    std::string output_name = output_info.begin()->first;
    // Load the network and create the inference request
    auto executable_network = core.LoadNetwork(network, argv[3]);
    const unsigned int ir_size = std::stoul(argv[4]);
    InferenceEngine::InferRequest::Ptr infer_request[ir_size];
    for (unsigned int ir = 0; ir < ir_size; ir++) {
        infer_request[ir] = executable_network.CreateInferRequestPtr();
    }
    const int BLOCK_SIZE = input_data->getTensorDesc().getDims().at(1) / 2;
    std::cout << "Block size: " << BLOCK_SIZE << std::endl;

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

    unsigned int ir_count = 0;
    unsigned int iters[ir_size];
    float res[n*2] = {};

    auto begin = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < n * 4; i += 4 * BLOCK_SIZE) {
        if (InferenceEngine::OK == infer_request[ir_count]->Wait(InferenceEngine::IInferRequest::WaitMode::RESULT_READY)) {
            // Create buffer to hold output data
            InferenceEngine::Blob::Ptr output = infer_request[ir_count]->GetBlob(output_name);
            float* output_buffer = output->buffer().as<InferenceEngine::PrecisionTrait<InferenceEngine::Precision::FP32>::value_type*>();
            for (int block = 0; block < BLOCK_SIZE; block++) {
                res[iters[ir_count] + block*2 + 0] = output_buffer[0+block*2];
                res[iters[ir_count] + block*2 + 1] = output_buffer[1+block*2];
            }
        }

        iters[ir_count] = i/2;
        // Create buffer to hold input data
        InferenceEngine::Blob::Ptr input = infer_request[ir_count]->GetBlob(input_name);
        float* input_buffer = input->buffer().as<InferenceEngine::PrecisionTrait<InferenceEngine::Precision::FP32>::value_type*>();
        for (int block = 0; block < BLOCK_SIZE; block++) {
            input_buffer[0+block*2]  = t1t2xy[i + block*4 + 2];
            input_buffer[1+block*2]  = t1t2xy[i + block*4 + 3];
        }
        infer_request[ir_count]->StartAsync();

        ir_count++;
        if (ir_count == ir_size) {
            ir_count = 0;
        }
    }
    for (unsigned int ir = 0; ir < ir_size; ir++) {
        if (InferenceEngine::OK == infer_request[ir]->Wait(InferenceEngine::IInferRequest::WaitMode::RESULT_READY)) {
            // Create buffer to hold output data
            InferenceEngine::Blob::Ptr output = infer_request[ir]->GetBlob(output_name);
            float* output_buffer = output->buffer().as<InferenceEngine::PrecisionTrait<InferenceEngine::Precision::FP32>::value_type*>();
            for (int block = 0; block < BLOCK_SIZE; block++) {
                res[iters[ir] + block*2 + 0] = output_buffer[0+block*2];
                res[iters[ir] + block*2 + 1] = output_buffer[1+block*2];
            }
        }
    }
    auto end = std::chrono::high_resolution_clock::now();
    std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(end-begin).count() << " ms" << std::endl;

    for (int i = 0; i < n * 2; i += 2) {
        outputFileHandler <<  res[i+0] << "\t" << res[i+1] << "\n";
    }

    inputFileHandler.close();
    outputFileHandler.close();

    free(t1t2xy);

    return 0;
}
