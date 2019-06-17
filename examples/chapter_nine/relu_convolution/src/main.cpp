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

Matrix convolution(const Matrix & input, const Matrix & filter)
{
    int filterRows = filter.rows();
    int filterCols = filter.cols();
    int rows = input.rows() - filterRows + 1;
    int cols = input.cols() - filterCols + 1;
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

std::tuple<Matrix, Matrix> imageConvolution(const Matrix &input, cv::Mat &dest, const Matrix & filter)
{
    const Matrix convoluted = convolution(input, filter);
    const Matrix relu = convoluted.unaryExpr([](double coeff){
        return std::max(0.0, coeff);
    });
    cv::Mat temp;
    cv::eigen2cv(relu, temp);
    temp.convertTo(dest, CV_8UC1);
    return std::make_tuple(convoluted, relu);
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
    auto [stdIgnore, groundTruth] = imageConvolution(X, groundTruthImage, groundTruthFilter);
    Matrix K = 0.05 * Matrix::Random(3, 3);
    int key = 0;
    int epoch = 0;
    while(key != 27)
    {
        auto [convoluted, output] = imageConvolution(X, outputImage, K);
        Matrix dC = output - groundTruth;
        double mse = dC.cwiseProduct(dC).sum() / (dC.rows() * dC.cols()); 
        const Matrix dRelu = dC.binaryExpr(convoluted, [](double _dC, double _convoluted){
            double result = 0.0;
            if(_convoluted > 0.0) result = _dC;
            return result;
        });
        Matrix dK = convolution(X, dRelu);
        std::cout << "Epoch:\t" << epoch << "\tMSE:\t" << mse << "\n";
        K = K - learningRate * dK;  
        show(image, groundTruthImage, outputImage);
        key = cv::waitKey(10);
        epoch++;
    }
    std::cout << "K = \n" << K << "\n\n";
    return 0;
}