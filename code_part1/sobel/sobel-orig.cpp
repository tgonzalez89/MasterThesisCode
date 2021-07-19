#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <chrono>
#include <iostream>

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
    if (s > 0.7071) {
        s = 0.7071;
    }
    return s;
}

int main(int argc, char* argv[]) {
    cv::Mat image = cv::imread(argv[1], cv::IMREAD_GRAYSCALE);
    cv::Mat image_float, grad_norm;
    cv::Mat grad = cv::Mat(image.rows, image.cols, CV_32F);
    image.convertTo(image_float, CV_32F, 1.0/255.0);

    auto begin = std::chrono::high_resolution_clock::now();
    cv::copyMakeBorder(image_float, image_float, 1, 1, 1, 1, cv::BORDER_DEFAULT);
    //#pragma omp parallel for
    for (int i=1; i <= image.rows; i++) {
        for (int j=1; j <= image.cols; j++) {
            float w[3][3] = {{image_float.at<float>(i-1,j-1),image_float.at<float>(i-1,j  ),image_float.at<float>(i-1,j+1)},
                             {image_float.at<float>(i  ,j-1),image_float.at<float>(i  ,j  ),image_float.at<float>(i  ,j+1)},
                             {image_float.at<float>(i+1,j-1),image_float.at<float>(i+1,j  ),image_float.at<float>(i+1,j+1)}};
            grad.at<float>(i-1, j-1) = sobel(w);
        }
    }
    auto end = std::chrono::high_resolution_clock::now();
    std::cout << std::chrono::duration_cast<std::chrono::microseconds>(end-begin).count() << " us" << std::endl;

    //cv::threshold(grad, grad, 3.0, 3.0, cv::THRESH_TRUNC);
    cv::normalize(grad, grad_norm, 0, 255, cv::NORM_MINMAX, CV_8U);
    cv::imwrite(argv[2], grad_norm);
    return 0;
}
