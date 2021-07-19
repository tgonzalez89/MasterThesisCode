#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <inference_engine.hpp>
#include <chrono>
#include <iostream>
#include <string>

int BLOCK_SIZE = 1;
#define KERNEL_SIZE 3
#define PADDING 1  // KERNEL_SIZE/2

int main(int argc, char* argv[]) {
    // Create IE core object and read the network
    InferenceEngine::Core core;
    InferenceEngine::CNNNetwork network = core.ReadNetwork("nn_sobel-2D.xml", "nn_sobel-2D.bin");
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
    unsigned int coords[ir_size][2];
    for (unsigned int ir = 0; ir < ir_size; ir++) {
        infer_request[ir] = executable_network.CreateInferRequestPtr();
    }

    BLOCK_SIZE = input_data->getTensorDesc().getDims().at(2) - 2;

    cv::Mat image = cv::imread(argv[1], cv::IMREAD_GRAYSCALE);
    cv::Mat image_float, grad_norm;
    image.convertTo(image_float, CV_32F, 1.0/255.0);

    auto begin = std::chrono::high_resolution_clock::now();
    int bottom_pad = image.rows%BLOCK_SIZE ? BLOCK_SIZE-image.rows%BLOCK_SIZE : 0;
    int right_pad =  image.cols%BLOCK_SIZE ? BLOCK_SIZE-image.cols%BLOCK_SIZE : 0;
    cv::copyMakeBorder(image_float, image_float, PADDING, bottom_pad+PADDING, PADDING, right_pad+PADDING, cv::BORDER_DEFAULT);
    cv::Mat grad = cv::Mat(image_float.rows, image_float.cols, CV_32F);
    unsigned int ir_count = 0;
    for (int i=PADDING; i < image.rows+PADDING; i+=BLOCK_SIZE) {
        for (int j=PADDING; j < image.cols+PADDING; j+=BLOCK_SIZE) {

            if (InferenceEngine::OK == infer_request[ir_count]->Wait(InferenceEngine::IInferRequest::WaitMode::RESULT_READY)) {
                // Create buffer to hold output data
                InferenceEngine::Blob::Ptr output = infer_request[ir_count]->GetBlob(output_name);
                float* output_buffer = output->buffer().as<InferenceEngine::PrecisionTrait<InferenceEngine::Precision::FP32>::value_type*>();
                cv::Mat grad_block = cv::Mat::zeros(BLOCK_SIZE, BLOCK_SIZE, CV_32F);
                for (int y = 0; y < BLOCK_SIZE; y++) {
                    for (int x = 0; x < BLOCK_SIZE; x++) {
                        grad_block.at<float>(y, x) = output_buffer[(y+PADDING)*(BLOCK_SIZE+PADDING*2) + (x+PADDING)];
                    }
                }
                grad_block.copyTo(grad(cv::Rect(coords[ir_count][1], coords[ir_count][0], grad_block.cols, grad_block.rows)));
            }

            coords[ir_count][0] = i;
            coords[ir_count][1] = j;
            // Create buffer to hold input data
            InferenceEngine::Blob::Ptr input = infer_request[ir_count]->GetBlob(input_name);
            float* input_buffer = input->buffer().as<InferenceEngine::PrecisionTrait<InferenceEngine::Precision::FP32>::value_type*>();
            cv::Mat block = image_float(cv::Rect(j-PADDING, i-PADDING, BLOCK_SIZE+PADDING*2, BLOCK_SIZE+PADDING*2));
            for (int y = 0; y < BLOCK_SIZE+PADDING*2; y++) {
                for (int x = 0; x < BLOCK_SIZE+PADDING*2; x++) {
                    input_buffer[y*(BLOCK_SIZE+PADDING*2) + x] = block.at<float>(y, x);
                }
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
            cv::Mat grad_block = cv::Mat::zeros(BLOCK_SIZE, BLOCK_SIZE, CV_32F);
            for (int y = 0; y < BLOCK_SIZE; y++) {
                for (int x = 0; x < BLOCK_SIZE; x++) {
                    grad_block.at<float>(y, x) = output_buffer[(y+PADDING)*(BLOCK_SIZE+PADDING*2) + (x+PADDING)];
                }
            }
            grad_block.copyTo(grad(cv::Rect(coords[ir][1], coords[ir][0], grad_block.cols, grad_block.rows)));
        }
    }
    grad = grad(cv::Rect(PADDING, PADDING, image.cols, image.rows));
    auto end = std::chrono::high_resolution_clock::now();
    std::cout << std::chrono::duration_cast<std::chrono::microseconds>(end-begin).count() << " us" << std::endl;

    //cv::threshold(grad, grad, 3.0, 3.0, cv::THRESH_TRUNC);
    cv::normalize(grad, grad_norm, 0, 255, cv::NORM_MINMAX, CV_8U);
    cv::imwrite(argv[2], grad_norm);
    return 0;
}
