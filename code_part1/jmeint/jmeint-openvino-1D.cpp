#include <fstream>
#include <iostream>
#include <chrono>
#include <cmath>
#include <string>
#include <inference_engine.hpp>

int main(int argc, char* argv[]) {
    // Create IE core object and read the network
    InferenceEngine::Core core;
    InferenceEngine::CNNNetwork network = core.ReadNetwork("nn_jmeint-1D.xml", "nn_jmeint-1D.bin");
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
    float* xyz = (float*)malloc(n * 6 * 3 * sizeof(float));
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

    const int BLOCK_SIZE = input_data->getTensorDesc().getDims().at(1) / 18;
    std::cout << "Block size: " << BLOCK_SIZE << std::endl;
    int res[n] = {};
    unsigned int ir_count = 0;
    unsigned int iters[ir_size];

    auto begin = std::chrono::high_resolution_clock::now();
    for (int i = 0 ; i < (n * 6 * 3); i += 6 * 3 * BLOCK_SIZE) {
        if (InferenceEngine::OK == infer_request[ir_count]->Wait(InferenceEngine::IInferRequest::WaitMode::RESULT_READY)) {
            // Create buffer to hold output data
            InferenceEngine::Blob::Ptr output = infer_request[ir_count]->GetBlob(output_name);
            float* output_buffer = output->buffer().as<InferenceEngine::PrecisionTrait<InferenceEngine::Precision::FP32>::value_type*>();
            for (int block = 0; block < BLOCK_SIZE; block++) {
                res[iters[ir_count] + block] = round(output_buffer[block]);
            }
        }

        iters[ir_count] = i / 18;
        // Create buffer to hold input data
        InferenceEngine::Blob::Ptr input = infer_request[ir_count]->GetBlob(input_name);
        float* input_buffer = input->buffer().as<InferenceEngine::PrecisionTrait<InferenceEngine::Precision::FP32>::value_type*>();
        for (int j = 0; j < (18 * BLOCK_SIZE); j++) {
            input_buffer[j]  = xyz[i + j];
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
                res[iters[ir] + block] = round(output_buffer[block]);
            }
        }
    }
    auto end = std::chrono::high_resolution_clock::now();
    std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(end-begin).count() << " ms" << std::endl;

    // Write output data
    for (int i = 0; i < n; i++) {
        outputFileHandler << res[i] << std::endl;
    }

    inputFileHandler.close();
    outputFileHandler.close();

    free(xyz);

    return 0;
}
