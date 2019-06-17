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

Matrix rotate180(const Matrix &m)
{
    return m.colwise().reverse().rowwise().reverse();
}

Matrix convolution(const Matrix &source, const Matrix &filter, const int row_padding, const int cols_padding)
{
    int filterRows = filter.rows();
    int filterCols = filter.cols();
    int rows = source.rows() - filterRows + 2*row_padding + 1;
    int cols = source.cols() - filterCols + 2*cols_padding + 1;
    Matrix input = Matrix::Zero(source.rows() + 2*row_padding, source.cols() + 2*cols_padding);
    input.block(row_padding, cols_padding, source.rows(), source.cols()) = source;

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

std::tuple<Matrix, Matrix> imageConvolution(const Matrix input, cv::Mat &dest, const Matrix &filterLayer1, const Matrix &filterLayer2)
{
    Matrix layer1 = convolution(input, filterLayer1, 0, 0);
    Matrix layer2 = convolution(layer1, filterLayer2, 0, 0);
    cv::eigen2cv(layer2, dest);
    dest.convertTo(dest, CV_8UC1);
    return std::make_tuple(layer1, layer2);
}

class MSEViewer {

    private:
        
        int height;
        int width;
        cv::Mat mat;
        double maxMSE;
        double maxEpoch;
        int bottom_padding;
        int top_padding;
        int left_padding;
        int right_padding;
    public:
        MSEViewer(int _height, int _width, double _maxMSE, double _maxEpoch, int _bottom_padding, int _top_padding, int _left_padding, int _right_padding) : 
            height(_height), width(_width), mat(cv::Mat(height, width, CV_8UC1, cv::Scalar(0))), maxMSE(_maxMSE), 
            maxEpoch(_maxEpoch), bottom_padding(_bottom_padding), top_padding(_top_padding), left_padding(_left_padding), right_padding(_right_padding) {
                int bottommost = height - bottom_padding;
                line(mat, cv::Point(left_padding, bottommost), cv::Point(left_padding, bottom_padding), cv::Scalar(255));
                line(mat, cv::Point(left_padding, bottommost), cv::Point(width - right_padding, bottommost), cv::Scalar(255));
            }
        void addPoint(int epoch, double mse) {
            if(epoch < maxEpoch)
            {
                double normMse = mse / maxMSE;
                int maxMSEPos = mat.rows - bottom_padding;
                int positionY = static_cast<int>(-normMse * (maxMSEPos - top_padding) + maxMSEPos);

                double normEpoch = epoch / maxEpoch;
                int maxEpochPos = mat.cols - right_padding;
                int positionX = static_cast<int>(normEpoch * (maxEpochPos - left_padding) + left_padding);

                cv::circle(mat, cv::Point(positionX, positionY), 2, cv::Scalar(255));
            }
        }
        const cv::Mat& getMat() const {
            return mat;
        }

};

int main(int, char **)
{
    const char * imagepath = "../data/convolution_example.png";
    cv::Mat image = cv::imread(imagepath, cv::IMREAD_GRAYSCALE);
    cv::Mat groundTruth, outputImage;
    Matrix groundTruthFilterLayer1(5, 5);
    groundTruthFilterLayer1 << 
    1, 4, 6, 4, 1, 
    4, 16, 24, 16, 4, 
    6, 24, 36, 24, 6,
    4, 16, 24, 16, 4, 
    1, 4, 6, 4, 1;
    groundTruthFilterLayer1 /= 256;
    Matrix groundTruthFilterLayer2(3, 3);
    groundTruthFilterLayer2 << 
    -1, 0, 1, 
    -1, 0, 1, 
    -1, 0, 1;
    Matrix inputMatrix;
    cv::cv2eigen(image, inputMatrix);
    const auto [stdIgnore, groundTruthMatrix] = imageConvolution(inputMatrix, groundTruth, groundTruthFilterLayer1, groundTruthFilterLayer2);
    Matrix K1 = 0.05 * Matrix::Random(5, 5);
    Matrix K2 = 0.05 * Matrix::Random(3, 3);
    int key = 0;
    int epoch = 0;
    double learningRate = 0.000000000001;
    while(key != 27)
    {
        auto [outputLayer1, outputLayer2] = imageConvolution(inputMatrix, outputImage, K1, K2);
        Matrix dCLayer2 = outputLayer2 - groundTruthMatrix;
        double mse = dCLayer2.cwiseProduct(dCLayer2).sum() / dCLayer2.rows() / dCLayer2.cols();

        Matrix dK2 = convolution(outputLayer1, dCLayer2, 0, 0);
        int row_padding = (outputLayer1.rows() - dCLayer2.rows() + K2.rows() - 1) / 2;
        int col_padding = (outputLayer1.cols() - dCLayer2.cols() + K2.cols() - 1) / 2;
        auto K2_180 = rotate180(K2);
        Matrix dCLayer1 = convolution(dCLayer2, K2_180, row_padding, col_padding);
        Matrix dK1 = convolution(inputMatrix, dCLayer1, 0, 0);
        K2 = K2 - learningRate * dK2;  
        K1 = K1 - learningRate * dK1; 
        std::cout << "Epoch:\t" << epoch << "\tMSE:\t" << mse << "\n";
        epoch++;
        show(image, groundTruth, outputImage);
        key = cv::waitKey(10);

    }
    std::cout << "K1 =\n" << K1*256 << "\n\n";
    std::cout << "K2 =\n" << K2 << "\n\n";
    return 0;
}