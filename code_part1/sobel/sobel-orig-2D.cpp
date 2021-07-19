#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <chrono>
#include <iostream>
#include <string>

float ky[3][3] = {{-1,-2,-1},
                  { 0, 0, 0},
                  { 1, 2, 1}};

float kx[3][3] = {{-1, 0, 1},
                  {-2, 0, 2},
                  {-1, 0, 1}};

float convolve(float w[3][3], float k[3][3]) {
    float r = 0;
    for (int i=0; i < 3; i++) {
        for (int j=0; j < 3; j++) {
            r += w[i][j] * k[i][j];
        }
    }
    return r;
}

float sobel(float w[3][3]) {
    float sy = convolve(w, ky);
    float sx = convolve(w, kx);
    float s = sqrt(sy * sy + sx * sx);
    return s;
}

int BLOCK_SIZE = 1;
#define KERNEL_SIZE 3
#define PADDING 1  // KERNEL_SIZE/2

cv::Mat sobel_block(cv::Mat block) {
    cv::Mat grad_block = cv::Mat::zeros(BLOCK_SIZE, BLOCK_SIZE, CV_32F);
    for (int i=PADDING; i <= BLOCK_SIZE; i++) {
        for (int j=PADDING; j <= BLOCK_SIZE; j++) {
            float w[3][3] = {{block.at<float>(i-1,j-1),block.at<float>(i-1,j  ),block.at<float>(i-1,j+1)},
                             {block.at<float>(i  ,j-1),block.at<float>(i  ,j  ),block.at<float>(i  ,j+1)},
                             {block.at<float>(i+1,j-1),block.at<float>(i+1,j  ),block.at<float>(i+1,j+1)}};
            grad_block.at<float>(i-PADDING, j-PADDING) = sobel(w);
        }
    }
    return grad_block;
}

int main(int argc, char* argv[]) {
    BLOCK_SIZE = std::stoi(argv[3], nullptr);

    cv::Mat image = cv::imread(argv[1], cv::IMREAD_GRAYSCALE);
    cv::Mat image_float, grad_norm;
    image.convertTo(image_float, CV_32F, 1.0/255.0);

    auto begin = std::chrono::high_resolution_clock::now();
    int bottom_pad = image.rows%BLOCK_SIZE ? BLOCK_SIZE-image.rows%BLOCK_SIZE : 0;
    int right_pad =  image.cols%BLOCK_SIZE ? BLOCK_SIZE-image.cols%BLOCK_SIZE : 0;
    cv::copyMakeBorder(image_float, image_float, PADDING, bottom_pad+PADDING, PADDING, right_pad+PADDING, cv::BORDER_DEFAULT);
    cv::Mat grad = cv::Mat(image_float.rows, image_float.cols, CV_32F);
    //#pragma omp parallel for
    for (int i=PADDING; i < image.rows+PADDING; i+=BLOCK_SIZE) {
        for (int j=PADDING; j < image.cols+PADDING; j+=BLOCK_SIZE) {
            cv::Mat block = image_float(cv::Rect(j-PADDING, i-PADDING, BLOCK_SIZE+PADDING*2, BLOCK_SIZE+PADDING*2));
            cv::Mat grad_block = sobel_block(block);
            grad_block.copyTo(grad(cv::Rect(j, i, grad_block.cols, grad_block.rows)));
        }
    }
    grad = grad(cv::Rect(PADDING, PADDING, image.cols, image.rows));
    auto end = std::chrono::high_resolution_clock::now();
    std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(end-begin).count() << " ms" << std::endl;

    cv::normalize(grad, grad_norm, 0, 255, cv::NORM_MINMAX, CV_8U);
    cv::imwrite(argv[2], grad_norm);
    return 0;
}
