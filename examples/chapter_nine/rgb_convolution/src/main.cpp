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

#include <unsupported/Eigen/CXX11/Tensor>

using Tensor3d = Eigen::Tensor<double, 3, Eigen::RowMajor>;

Tensor3d convertToTensor3d(const cv::Mat &image)
{
    const int rows = image.rows;
    const int cols = image.cols;
    cv::Mat reshaped = image.reshape(1, 3*cols*rows);
    Matrix temp;
    cv::cv2eigen(reshaped, temp);
    Tensor3d result = Eigen::TensorMap<Tensor3d>(temp.data(), rows, cols, 3);
    return result;
}

void show(const cv::Mat &source, const cv::Mat &groundTruth, const cv::Mat &convoluted)
{
    int maxHeight = std::max(std::max(source.rows, groundTruth.rows), convoluted.rows);
    int width = source.cols + groundTruth.cols + convoluted.cols;
    cv::Mat toShow(maxHeight, width, CV_8UC3, cv::Scalar(0, 0, 0));
    cv::Mat source_roi = toShow(cv::Rect(0, 0, source.cols, source.rows));
    source.copyTo(source_roi);
    cv::Mat groundTruth_roi = toShow(cv::Rect(source.cols, 0, groundTruth.cols, groundTruth.rows));
    cv::Mat groundTruth_rgb;
    if(groundTruth.channels() == 1)
    {
        cv::Mat temp;
        groundTruth.convertTo(temp, CV_8UC1);
        cv::cvtColor(temp, groundTruth_rgb, cv::COLOR_GRAY2BGR);
    } else
        groundTruth_rgb = groundTruth;
    groundTruth_rgb.copyTo(groundTruth_roi);
    cv::Mat convoluted_roi = toShow(cv::Rect(source.cols + groundTruth.cols, 0, convoluted.cols, convoluted.rows));
    cv::Mat convoluted_rgb;
    if(convoluted.channels() == 1)
    {
        cv::Mat temp;
        convoluted.convertTo(temp, CV_8UC1);
        cv::cvtColor(temp, convoluted_rgb, cv::COLOR_GRAY2BGR);
    } else
        convoluted_rgb = convoluted;
    convoluted_rgb.copyTo(convoluted_roi);
    cv::namedWindow("", cv::WindowFlags::WINDOW_AUTOSIZE);
    cv::imshow("", toShow);
}

Tensor3d convolution(const Tensor3d &input, const Tensor3d &filter)
{
    int filterRows = filter.dimension(0);
    int filterCols = filter.dimension(1);
    int filterChannels = filter.dimension(2);
    int rows = input.dimension(0) - filterRows + 1;
    int cols = input.dimension(1) - filterCols + 1;
    int channels = input.dimension(2) / filterChannels;
    Tensor3d result(rows, cols, channels);
    result.setConstant(0.0);
    Eigen::array<int, 3> offsets;
    Eigen::array<int, 3> extents;
    extents[0] = filterRows;
    extents[1] = filterCols;
    extents[2] = filterChannels;
    for(int i = 0; i < rows; ++i)
    {
        offsets[0] = i;
        for(int j = 0; j < cols; ++j)
        {
            offsets[1] = j;
            for(int k = 0; k < channels; ++k)
            {
                offsets[2] = k;
                Eigen::Tensor<double, 0, Eigen::RowMajor> sumProduct = (input.slice(offsets, extents) * filter).sum();
                double summ = sumProduct(0);
                result(i, j, k) = summ;
            }
        }
    }
    return result;
}

Tensor3d imageConvolution(const Tensor3d &input, cv::Mat &dest, const Tensor3d &filter)
{
    Tensor3d convoluted = convolution(input, filter);
    std::vector<cv::Mat> channels;
    int rows = convoluted.dimension(0);
    int cols = convoluted.dimension(1);
    int numberOfChannels = convoluted.dimension(2);
    channels.reserve(numberOfChannels);

    for(int k = 0; k < numberOfChannels; ++k)
    {
        Eigen::Tensor<double, 2, Eigen::RowMajor> chip = convoluted.chip(k, 2);
        Matrix matrix = Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> (chip.data(), rows, cols);
        cv::Mat temp, temp2;
        cv::eigen2cv(matrix, temp);
        temp.convertTo(temp2, CV_8UC1);
        channels.push_back(temp);
    }
    cv::merge(channels, dest);
    if(numberOfChannels == 1)
        dest.convertTo(dest, CV_8UC1);
    return convoluted;
}

int main(int, char **)
{

    const char * imagepath = "../data/convolution_example.png";
    const cv::Mat image = cv::imread(imagepath, cv::IMREAD_COLOR);
    cv::Mat groundTruth, outputImage;
    Tensor3d inputTensor = convertToTensor3d(image);

    Tensor3d groundTruthFilter(3, 3, 3);
    groundTruthFilter.setValues
        ({
            {{-1.0, -1.0, -1.0}, {0.0, 0.0, 0.0}, {1.0, 1.0, 1.0}},
            {{-1.0, -1.0, -1.0}, {0.0, 0.0, 0.0}, {1.0, 1.0, 1.0}},
            {{-1.0, -1.0, -1.0}, {0.0, 0.0, 0.0}, {1.0, 1.0, 1.0}}
        });

    Tensor3d K(3, 3, 3);
    K.setRandom();
    K = K * 0.05;
    const auto groundTruthTensor = imageConvolution(inputTensor, groundTruth, groundTruthFilter);
    int key = 0;
    int epoch = 0;
    double learningRate = 0.000000000001;
    while(key != 27)
    {
        Tensor3d convolutedImageTensor = imageConvolution(inputTensor, outputImage, K);
        
        Tensor3d dC = convolutedImageTensor - groundTruthTensor;
        Eigen::Tensor<double, 0, Eigen::RowMajor> mseTensor = (dC * dC).sum();
        double mse = mseTensor(0) / (dC.dimension(0) * dC.dimension(1) * dC.dimension(2));

        Tensor3d dK = convolution(inputTensor, dC);

        K = K - learningRate * dK;  

        std::cout << "Epoch:\t" << epoch << "\tMSE:\t" << mse << "\n";
        show(image, groundTruth, outputImage);
        epoch++;
        key = cv::waitKey(50);

    }
    std::cout << "K = \n" << K << "\n\n";
    return 0;
}