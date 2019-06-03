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

void show(const char *title, const cv::Mat &image)
{
    
    cv::namedWindow(title, cv::WindowFlags::WINDOW_AUTOSIZE);
    cv::imshow(title, image);
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

void imageConvolution(const cv::Mat source, cv::Mat &dest, const Matrix & filter)
{
    Matrix input;
    cv::cv2eigen(source, input);

    Matrix convoluted = convolution(input, filter);

    cv::Mat temp;
    cv::eigen2cv(convoluted, temp);
    temp.convertTo(dest, CV_8UC1);
}
 
int main(int, char **)
{
    const char * imagepath = "../data/convolution_example.png";
    cv::Mat image = cv::imread(imagepath, cv::IMREAD_GRAYSCALE);
    cv::Mat dest;
    Matrix filter(3, 3);
    //filter << -1, 0, 1, -1, 0, 1, -1, 0, 1;
    //filter << -1, -1, -1, 0, 0, 0, 1, 1, 1;
    filter << 0, -1, 0, -1, 5, -1, 0, -1, 0;
    imageConvolution(image, dest, filter);
    show("origin", image);
    show("dest", dest);
    cv::waitKey(0);
    return 0;
}