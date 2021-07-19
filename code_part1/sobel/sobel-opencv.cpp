#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <chrono>
#include <iostream>

int main(int argc, char* argv[]) {
    cv::Mat image = cv::imread(argv[1], cv::IMREAD_GRAYSCALE);
    cv::Mat image_float, grad_x, grad_y, grad, grad_norm;
    image.convertTo(image_float, CV_32F, 1.0/255.0);

    auto begin = std::chrono::high_resolution_clock::now();
    cv::Sobel(image_float, grad_x, CV_32F, 1, 0, 3);
    cv::Sobel(image_float, grad_y, CV_32F, 0, 1, 3);
    cv::magnitude(grad_x, grad_y, grad);
    auto end = std::chrono::high_resolution_clock::now();
    std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(end-begin).count() << " ms" << std::endl;

    cv::normalize(grad, grad_norm, 0, 255, cv::NORM_MINMAX, CV_8U);
    cv::imwrite(argv[2], grad_norm);
    return 0;
}
