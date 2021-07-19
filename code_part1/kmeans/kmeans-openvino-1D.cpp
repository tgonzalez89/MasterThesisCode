#include <vector>
#include <cstdlib>
#include <cmath>
#include <algorithm>
#include <chrono>
#include <iostream>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <inference_engine.hpp>


// TODO: Allow the user to provide the number of clusters.
#define NUM_CLUSTERS 6
#define SCALE 1


struct Centroid {
    float r = 0.0;
    float g = 0.0;
    float b = 0.0;
};


float euclideanDistance(cv::Vec3f p, Centroid c) {
    float r = 0.0;
    r += (p[0] - c.b) * (p[0] - c.b);
    r += (p[1] - c.g) * (p[1] - c.g);
    r += (p[2] - c.r) * (p[2] - c.r);
    return sqrt(r);
}


int main(int argc, char* argv[]) {
    // Create IE core object and read the network
    InferenceEngine::Core core;
    InferenceEngine::CNNNetwork network = core.ReadNetwork("nn_kmeans-1D.xml", "nn_kmeans-1D.bin");
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
    const int ir_size = std::stoul(argv[4]);
    InferenceEngine::InferRequest::Ptr infer_request[ir_size];
    float* input_buffer[ir_size];
    float* output_buffer[ir_size];
    for (int ir = 0; ir < ir_size; ir++) {
        infer_request[ir] = executable_network.CreateInferRequestPtr();
        // Create buffer to hold input data
        InferenceEngine::Blob::Ptr input = infer_request[ir]->GetBlob(input_name);
        input_buffer[ir] = input->buffer().as<InferenceEngine::PrecisionTrait<InferenceEngine::Precision::FP32>::value_type*>();
        // Create buffer to hold output data
        InferenceEngine::Blob::Ptr output = infer_request[ir]->GetBlob(output_name);
        output_buffer[ir] = output->buffer().as<InferenceEngine::PrecisionTrait<InferenceEngine::Precision::FP32>::value_type*>();
    }
    const int BLOCK_SIZE = input_data->getTensorDesc().getDims().at(1) / 6;
    std::cout << "Block size: " << BLOCK_SIZE << std::endl;

    // Read input
    cv::Mat input_img = cv::imread(argv[1], cv::IMREAD_COLOR);
    cv::Mat img;
    input_img.convertTo(img, CV_32F, 1.0/255.0);

    // Initialize clusters.
    // Initialize to a repeatable random value between 0 and 1.
    // TODO: Allow the user to provide their own values (read them from a file).
    std::vector<Centroid> clusters(NUM_CLUSTERS, Centroid());
    for (int c = 0; c < NUM_CLUSTERS; c++) {
        clusters[c].r = (float)SCALE * rand() / RAND_MAX;
        clusters[c].g = (float)SCALE * rand() / RAND_MAX;
        clusters[c].b = (float)SCALE * rand() / RAND_MAX;
    }

    // Do the segmentation.
    auto begin = std::chrono::high_resolution_clock::now();
    std::vector<std::vector<std::vector<float>>> distances(img.rows, std::vector<std::vector<float>>(img.cols, std::vector<float>(NUM_CLUSTERS, 0.0)));
    cv::Mat assigned_clusters(img.rows, img.cols, CV_8U);
    int ir_count = 0;
    int block_count = 0;
    int iters[ir_size][BLOCK_SIZE][3] = {0};
    for (int i = 0; i < img.rows; i++) {
        for (int j = 0; j < img.cols; j++) {
            for (int c = 0; c < NUM_CLUSTERS; c++) {
                iters[ir_count][block_count][0] = i;
                iters[ir_count][block_count][1] = j;
                iters[ir_count][block_count][2] = c;
                input_buffer[ir_count][0+(block_count*6)] = img.at<cv::Vec3f>(i, j)[0];
                input_buffer[ir_count][1+(block_count*6)] = img.at<cv::Vec3f>(i, j)[1];
                input_buffer[ir_count][2+(block_count*6)] = img.at<cv::Vec3f>(i, j)[2];
                input_buffer[ir_count][3+(block_count*6)] = clusters[c].b;
                input_buffer[ir_count][4+(block_count*6)] = clusters[c].g;
                input_buffer[ir_count][5+(block_count*6)] = clusters[c].r;

                block_count++;
                if (block_count == BLOCK_SIZE) {
                    infer_request[ir_count]->StartAsync();
                    ir_count++;
                    if (ir_count == ir_size) {
                        ir_count = 0;
                    }
                    if (InferenceEngine::OK == infer_request[ir_count]->Wait(InferenceEngine::IInferRequest::WaitMode::RESULT_READY)) {
                        for (int block = 0; block < BLOCK_SIZE; block++) {
                            int _i = iters[ir_count][block][0];
                            int _j = iters[ir_count][block][1];
                            int _c = iters[ir_count][block][2];
                            distances[_i][_j][_c] = output_buffer[ir_count][block];
                            if (_c+1 == NUM_CLUSTERS) {
                                assigned_clusters.at<unsigned char>(_i, _j) = std::min_element(distances[_i][_j].begin(), distances[_i][_j].end()) - distances[_i][_j].begin();
                            }
                        }
                    }
                    block_count = 0;
                }
            }
        }
    }
    if (block_count != 0) {
        infer_request[ir_count]->StartAsync();
    }
    for (int ir = 0; ir < ir_size; ir++) {
        if (InferenceEngine::OK == infer_request[ir]->Wait(InferenceEngine::IInferRequest::WaitMode::RESULT_READY)) {
            for (int block = 0; block < BLOCK_SIZE; block++) {
                int _i = iters[ir][block][0];
                int _j = iters[ir][block][1];
                int _c = iters[ir][block][2];
                distances[_i][_j][_c] = output_buffer[ir][block];
                if (_c+1 == NUM_CLUSTERS) {
                    assigned_clusters.at<unsigned char>(_i, _j) = std::min_element(distances[_i][_j].begin(), distances[_i][_j].end()) - distances[_i][_j].begin();
                }
            }
        }
    }
    auto end = std::chrono::high_resolution_clock::now();
    std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(end-begin).count() << " ms" << std::endl;

    // Re-assign the average RGB value in each cluster to each pixel in that cluster.
    // 1) Reset cluster RGB values to 0.
    for (int c = 0; c < NUM_CLUSTERS; c++) {
        clusters[c].r = 0.0;
        clusters[c].g = 0.0;
        clusters[c].b = 0.0;
    }
    // 2) Add up all the RGB values for all the pixels in each cluster.
    int pixels_in_cluster[NUM_CLUSTERS] = {0};
    for (int i = 0; i < img.rows; i++) {
        for (int j = 0; j < img.cols; j++) {
            clusters[assigned_clusters.at<unsigned char>(i, j)].b += img.at<cv::Vec3f>(i, j)[0];
            clusters[assigned_clusters.at<unsigned char>(i, j)].g += img.at<cv::Vec3f>(i, j)[1];
            clusters[assigned_clusters.at<unsigned char>(i, j)].r += img.at<cv::Vec3f>(i, j)[2];
            pixels_in_cluster[assigned_clusters.at<unsigned char>(i, j)]++;
        }
    }
    // 3) Average the RGB values for all the pixels in each cluster.
    for (int c = 0; c < NUM_CLUSTERS; c++) {
        clusters[c].r = clusters[c].r / pixels_in_cluster[c];
        clusters[c].g = clusters[c].g / pixels_in_cluster[c];
        clusters[c].b = clusters[c].b / pixels_in_cluster[c];
    }
    // 4) Assign the averaged RGB values to each pixel in the image.
    for (int i = 0; i < img.rows; i++) {
        for (int j = 0; j < img.cols; j++) {
            img.at<cv::Vec3f>(i, j) = cv::Vec3f(
                clusters[assigned_clusters.at<unsigned char>(i, j)].b,
                clusters[assigned_clusters.at<unsigned char>(i, j)].g,
                clusters[assigned_clusters.at<unsigned char>(i, j)].r
            );
        }
    }

    // Write ouput
    cv::Mat output_img;
    img.convertTo(output_img, CV_8U, 255);
    cv::imwrite(argv[2], output_img);
    return 0;
}
