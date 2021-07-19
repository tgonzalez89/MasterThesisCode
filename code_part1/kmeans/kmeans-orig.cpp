#include <vector>
#include <cstdlib>
#include <cmath>
#include <algorithm>
#include <chrono>
#include <iostream>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>


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
    std::vector<float> distances(NUM_CLUSTERS, 0.0);
    cv::Mat assigned_clusters(img.rows, img.cols, CV_8U);
    for (int i = 0; i < img.rows; i++) {
        for (int j = 0; j < img.cols; j++) {
            // Calculate the euclidean distance to each of the clusters.
            for (int c = 0; c < NUM_CLUSTERS; c++) {
                distances[c] = euclideanDistance(img.at<cv::Vec3f>(i, j), clusters[c]);
            }
            // Assign the correct cluster based on the shortest euclidean distance.
            assigned_clusters.at<unsigned char>(i, j) = std::min_element(distances.begin(), distances.end()) - distances.begin();
        }
    }
    auto end = std::chrono::high_resolution_clock::now();
    std::cout << std::chrono::duration_cast<std::chrono::microseconds>(end-begin).count() << " us" << std::endl;

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

    // Write output
    cv::Mat output_img;
    img.convertTo(output_img, CV_8U, 255);
    cv::imwrite(argv[2], output_img);
    return 0;
}
