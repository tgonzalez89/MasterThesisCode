#include <cstdio>
#include <cmath>
#include <iostream>
#include <string>
#include <chrono>
#include <inference_engine.hpp>


#define DIVIDE 120.0
#define fptype float  // Precision to use for calculations
#define NUM_RUNS 1
#define PAD 256
#define LINESIZE 64


typedef struct OptionData_ {
        fptype s;          // spot price
        fptype strike;     // strike price
        fptype r;          // risk-free interest rate
        fptype divq;       // dividend rate
        fptype v;          // volatility
        fptype t;          // time to maturity or option expiration in years
                           //     (1yr = 1.0, 6mos = 0.5, 3mos = 0.25, ..., etc)
        char OptionType;   // Option type.  "P"=PUT, "C"=CALL
        fptype divs;       // dividend vals (not used in this test)
        fptype DGrefval;   // DerivaGem Reference Value
} OptionData;

OptionData* data;
fptype* prices;
int numOptions;

int*    otype;
fptype* sptprice;
fptype* strike;
fptype* rate;
fptype* volatility;
fptype* otime;


int bs_thread(void* tid_ptr, InferenceEngine::InferRequest::Ptr* infer_request, const unsigned int ir_size, const int BLOCK_SIZE, std::string input_name, std::string output_name) {
    int i, j;
    int tid = *(int*)tid_ptr;
    int start = tid * (numOptions);
    int end = start + (numOptions);

    unsigned int ir_count = 0;
    unsigned int iters[ir_size];

    for (j = 0; j < NUM_RUNS; j++) {
        for (i = start; i < end; i += BLOCK_SIZE) {
            if (InferenceEngine::OK == infer_request[ir_count]->Wait(InferenceEngine::IInferRequest::WaitMode::RESULT_READY)) {
                // Create buffer to hold output data
                InferenceEngine::Blob::Ptr output = infer_request[ir_count]->GetBlob(output_name);
                float* output_buffer = output->buffer().as<InferenceEngine::PrecisionTrait<InferenceEngine::Precision::FP32>::value_type*>();
                for (int block = 0; block < BLOCK_SIZE; block++) {
                    prices[iters[ir_count] + block] = output_buffer[block];
                }
            }

            iters[ir_count] = i;
            // Create buffer to hold input data
            InferenceEngine::Blob::Ptr input = infer_request[ir_count]->GetBlob(input_name);
            float* input_buffer = input->buffer().as<InferenceEngine::PrecisionTrait<InferenceEngine::Precision::FP32>::value_type*>();
            for (int block = 0; block < BLOCK_SIZE; block++) {
                input_buffer[0+(block*6)] = sptprice[i+block];
                input_buffer[1+(block*6)] = strike[i+block];
                input_buffer[2+(block*6)] = rate[i+block];
                input_buffer[3+(block*6)] = volatility[i+block];
                input_buffer[4+(block*6)] = otime[i+block];
                input_buffer[5+(block*6)] = otype[i+block];
            }
            infer_request[ir_count]->StartAsync();

            ir_count++;
            if (ir_count == ir_size) {
                ir_count = 0;
            }
        }
    }
    for (unsigned int ir = 0; ir < ir_size; ir++) {
        if (InferenceEngine::OK == infer_request[ir]->Wait(InferenceEngine::IInferRequest::WaitMode::RESULT_READY)) {
            // Create buffer to hold output data
            InferenceEngine::Blob::Ptr output = infer_request[ir]->GetBlob(output_name);
            float* output_buffer = output->buffer().as<InferenceEngine::PrecisionTrait<InferenceEngine::Precision::FP32>::value_type*>();
            for (int block = 0; block < BLOCK_SIZE; block++) {
                prices[iters[ir] + block] = output_buffer[block];
            }
        }
    }

    return 0;
}


int main (int argc, char* argv[]) {
    // Create IE core object and read the network
    InferenceEngine::Core core;
    InferenceEngine::CNNNetwork network = core.ReadNetwork("nn_blackscholes-1D.xml", "nn_blackscholes-1D.bin");
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

    FILE *file;
    int i;
    int loopnum;
    fptype* buffer;
    int* buffer2;
    int rv;

    fflush(NULL);

    char* inputFile = argv[1];
    char* outputFile = argv[2];

    //Read input data from file
    file = fopen(inputFile, "r");
    if (file == NULL) {
        printf("ERROR: Unable to open file `%s'.\n", inputFile);
        exit(1);
    }
    rv = fscanf(file, "%i", &numOptions);
    if (rv != 1) {
        printf("ERROR: Unable to read from file `%s'.\n", inputFile);
        fclose(file);
        exit(1);
    }

    // Alloc spaces for the option data
    data = (OptionData*)malloc(numOptions * sizeof(OptionData));
    prices = (fptype*)malloc(numOptions * sizeof(fptype));
    for (loopnum = 0; loopnum < numOptions; ++ loopnum) {
        rv = fscanf(file, "%f %f %f %f %f %f %c %f %f", &data[loopnum].s, &data[loopnum].strike, &data[loopnum].r,
                                                        &data[loopnum].divq, &data[loopnum].v, &data[loopnum].t,
                                                        &data[loopnum].OptionType, &data[loopnum].divs, &data[loopnum].DGrefval);
        if (rv != 9) {
            printf("ERROR: Unable to read from file `%s'.\n", inputFile);
            fclose(file);
            exit(1);
        }
    }
    rv = fclose(file);
    if (rv != 0) {
        printf("ERROR: Unable to close file `%s'.\n", inputFile);
        exit(1);
    }

    buffer = (fptype*)malloc(5 * numOptions * sizeof(fptype) + PAD);
    sptprice = (fptype*)(((unsigned long long)buffer + PAD) & ~(LINESIZE - 1));
    strike = sptprice + numOptions;
    rate = strike + numOptions;
    volatility = rate + numOptions;
    otime = volatility + numOptions;

    buffer2 = (int*)malloc(numOptions * sizeof(fptype) + PAD);
    otype = (int*)(((unsigned long long)buffer2 + PAD) & ~(LINESIZE - 1));

    for (i = 0; i < numOptions; i++) {
        otype[i]      = (data[i].OptionType == 'P') ? 1 : 0;
        sptprice[i]   = data[i].s / DIVIDE;
        strike[i]     = data[i].strike / DIVIDE;
        rate[i]       = data[i].r;
        volatility[i] = data[i].v;
        otime[i]      = data[i].t;
    }

    // Serial version
    const int BLOCK_SIZE = input_data->getTensorDesc().getDims().at(1) / 6;
    std::cout << "Block size: " << BLOCK_SIZE << std::endl;
    int tid = 0;
    auto begin = std::chrono::high_resolution_clock::now();
    bs_thread(&tid, infer_request, ir_size, BLOCK_SIZE, input_name, output_name);
    auto end = std::chrono::high_resolution_clock::now();
    std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(end-begin).count() << " ms" << std::endl;

    // Write prices to output file
    file = fopen(outputFile, "w");
    if (file == NULL) {
        printf("ERROR: Unable to open file `%s'.\n", outputFile);
        exit(1);
    }
    if (rv < 0) {
        printf("ERROR: Unable to write to file `%s'.\n", outputFile);
        fclose(file);
        exit(1);
    }
    for (i = 0; i < numOptions; i++) {
        rv = fprintf(file, "%.18f\n", prices[i]);
        if (rv < 0) {
            printf("ERROR: Unable to write to file `%s'.\n", outputFile);
            fclose(file);
            exit(1);
        }
    }
    rv = fclose(file);
    if (rv != 0) {
        printf("ERROR: Unable to close file `%s'.\n", outputFile);
        exit(1);
    }

    free(data);
    free(prices);

    return 0;
}
