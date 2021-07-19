#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <inference_engine.hpp>
#include <chrono>
#include <iostream>
#include <string>

int main(int argc, char* argv[]) {
    // Create IE core object and read the network
    InferenceEngine::Core core;
    InferenceEngine::CNNNetwork network = core.ReadNetwork("nn_sobel-0D.xml", "nn_sobel-0D.bin");
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

    cv::Mat image = cv::imread(argv[1], cv::IMREAD_GRAYSCALE);
    cv::Mat image_float, grad_norm;
    cv::Mat grad = cv::Mat(image.rows, image.cols, CV_32F);
    image.convertTo(image_float, CV_32F, 1.0/255.0);

    auto begin = std::chrono::high_resolution_clock::now();
    cv::copyMakeBorder(image_float, image_float, 1, 1, 1, 1, cv::BORDER_DEFAULT);
    unsigned int ir_count = 0;
    for (int i=1; i <= image.rows; i++) {
        for (int j=1; j <= image.cols; j++) {

            if (InferenceEngine::OK == infer_request[ir_count]->Wait(InferenceEngine::IInferRequest::WaitMode::RESULT_READY)) {
                // Create buffer to hold output data
                InferenceEngine::Blob::Ptr output = infer_request[ir_count]->GetBlob(output_name);
                float* output_buffer = output->buffer().as<InferenceEngine::PrecisionTrait<InferenceEngine::Precision::FP32>::value_type*>();
                grad.at<float>(coords[ir_count][0]-1, coords[ir_count][1]-1) = output_buffer[0];
            }

            coords[ir_count][0] = i;
            coords[ir_count][1] = j;
            // Create buffer to hold input data
            InferenceEngine::Blob::Ptr input = infer_request[ir_count]->GetBlob(input_name);
            float* input_buffer = input->buffer().as<InferenceEngine::PrecisionTrait<InferenceEngine::Precision::FP32>::value_type*>();
            input_buffer[0] = image_float.at<float>(i-1,j-1);
            input_buffer[1] = image_float.at<float>(i-1,j  );
            input_buffer[2] = image_float.at<float>(i-1,j+1);
            input_buffer[3] = image_float.at<float>(i  ,j-1);
            input_buffer[4] = image_float.at<float>(i  ,j  );
            input_buffer[5] = image_float.at<float>(i  ,j+1);
            input_buffer[6] = image_float.at<float>(i+1,j-1);
            input_buffer[7] = image_float.at<float>(i+1,j  );
            input_buffer[8] = image_float.at<float>(i+1,j+1);
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
            grad.at<float>(coords[ir][0]-1, coords[ir][1]-1) = output_buffer[0];
        }
    }
    auto end = std::chrono::high_resolution_clock::now();
    std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(end-begin).count() << " ms" << std::endl;

    cv::normalize(grad, grad_norm, 0, 255, cv::NORM_MINMAX, CV_8U);
    cv::imwrite(argv[2], grad_norm);
    return 0;
}
