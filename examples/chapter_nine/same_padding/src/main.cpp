#include <iostream>
#include <fstream>
#include <exception>
#include <string>
#include <algorithm>

#include "matrix_definitions.hpp"

#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/eigen.hpp>

void show(const cv::Mat &source, const cv::Mat &groundTruth, const cv::Mat &convoluted)
{
    int maxHeight = std::max(std::max(source.rows, groundTruth.rows), convoluted.rows);
    int width = source.cols + groundTruth.cols + convoluted.cols;
    cv::Mat toShow(maxHeight, width, CV_8UC1, cv::Scalar(0));
    cv::Mat source_roi = toShow(cv::Rect(0, 0, source.cols, source.rows));
    source.copyTo(source_roi);
    cv::Mat groundTruth_roi = toShow(cv::Rect(source.cols, 0, groundTruth.cols, groundTruth.rows));
    groundTruth.copyTo(groundTruth_roi);
    cv::Mat convoluted_roi = toShow(cv::Rect(source.cols + groundTruth.cols, 0, convoluted.cols, convoluted.rows));
    convoluted.copyTo(convoluted_roi);
    cv::namedWindow("", cv::WindowFlags::WINDOW_AUTOSIZE);
    cv::imshow("", toShow);
}

Matrix convolution(const Matrix & source, const Matrix & filter, const int padding)
{
    int filterRows = filter.rows();
    int filterCols = filter.cols();
    int rows = source.rows() - filterRows + 2*padding + 1;
    int cols = source.cols() - filterCols + 2*padding + 1;
    Matrix input = Matrix::Zero(source.rows() + 2*padding, source.cols() + 2*padding);
    input.block(padding, padding, source.rows(), source.cols()) = source;

    Matrix result = Matrix::Zero(rows, cols);
    for(int i = 0; i < rows; ++i)
    {
        for(int j = 0; j < cols; ++j)
        {
            double sum = input.block(i, j, filterRows, filterCols).cwiseProduct(filter).sum();
            result(i, j) = sum;
        }
    }
    return result;
}

Matrix imageConvolution(const Matrix &input, cv::Mat &dest, const Matrix & filter, const int padding)
{
    Matrix convoluted = convolution(input, filter, padding);
    cv::Mat temp;
    cv::eigen2cv(convoluted, temp);
    temp.convertTo(dest, CV_8UC1);
    return convoluted;
}

int main(int, char **)
{
    srand(0);
    const char * imagepath = "../data/convolution_example.png";
    cv::Mat image = cv::imread(imagepath, cv::IMREAD_GRAYSCALE);
    cv::Mat groundTruthImage, outputImage;
    Matrix groundTruthFilter(3, 3);
    groundTruthFilter << 
        -1, 0, 1, 
        -1, 0, 1, 
        -1, 0, 1;
    double learningRate = 0.00000000001;
    Matrix X;
    cv::cv2eigen(image, X);
    const int padding = (groundTruthFilter.rows() - 1) / 2;
    const Matrix groundTruth = imageConvolution(X, groundTruthImage, groundTruthFilter, padding);
    Matrix K = 0.05 * Matrix::Random(3, 3);
    int key = 0;
    int epoch = 0;
    while(key != 27)
    {
        Matrix output = imageConvolution(X, outputImage, K, padding);
        Matrix dC = output - groundTruth;
        double mse = dC.cwiseProduct(dC).sum() / (dC.rows() * dC.cols()); 
        Matrix dK = convolution(X, dC, padding);
        std::cout << "Epoch:\t" << epoch << "\tMSE:\t" << mse << "\n";
        K = K - learningRate * dK;  
        show(image, groundTruthImage, outputImage);
        key = cv::waitKey(10);
        epoch++;
    }
    std::cout << "K = \n" << K << "\n\n";
    return 0;
}