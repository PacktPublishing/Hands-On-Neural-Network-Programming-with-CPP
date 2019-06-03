#include <iostream>
#include <fstream>
#include <exception>
#include <string>
#include <algorithm>

#include "matrix_definitions.hpp"

Matrix convolution(const Matrix & input, const Matrix & filter)
{
    int filterRows = filter.rows();
    int filterCols = filter.cols();
    int rows = (input.rows() - filterRows) + 1;
    int cols = (input.cols() - filterCols) + 1;
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

int main(int, char **)
{
Matrix filter(3, 3);
filter << -1, 0, 1, 
            -1, 0, 1, 
            -1, 0, 1;
std::cout << filter << "\n\n";
Matrix input(6, 6);
input << 3, 1, 0, 2, 5, 6, 
            4, 2, 1, 1, 4, 7, 
            5, 4, 0, 0, 1, 2, 
            1, 2, 2, 1, 3, 4,
            6, 3, 1, 0, 5, 2,
            3, 1, 0, 1, 3, 3;
std::cout << input << "\n\n";
Matrix output = convolution(input, filter);
    std::cout << output << "\n";
    return 0;
}