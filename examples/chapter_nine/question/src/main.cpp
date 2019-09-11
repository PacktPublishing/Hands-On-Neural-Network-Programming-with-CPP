#include <iostream>
#include "matrix_definitions.hpp"

/**
* schoolbook version of basic convolution (no padding, no strides)
*/ 
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

/**
* Output of ReLU
*/
Matrix ReLU(const Matrix & input) 
{
    const Matrix result = input.unaryExpr([](double coeff){
        return std::max(0.0, coeff);
    });
    return result;
}

/**
* Output of Max Pooling
*/
Matrix maxPooling(const Matrix & input, const int pooling, int strides) 
{
    int rows = (input.rows() - pooling) / strides + 1;
    int cols = (input.cols() - pooling) / strides + 1;
    Matrix result = Matrix::Zero(rows, cols);
    for(int i = 0; i < rows; i = i + 1)
    {
        for(int j = 0; j < cols; j = j + 1)
        {
            double maxValue = input.block(strides*i, strides*j, pooling, pooling).maxCoeff();
            result(i, j) = maxValue;
        }
    }
    return result;
}

/**
* Evaluation of quadratic cost
*/
double quadraticCost(const Matrix & output, const Matrix & expectedOutput) 
{
    Matrix diff = output - expectedOutput;
    double result = diff.cwiseProduct(diff).sum() / (diff.rows() * diff.cols()) / 2.0; 
    return result;
}

/**
* This function is a stub. Implement it to return the quadratic cost gradient
*/ 
Matrix quadraticCostGradient(const Matrix & output, const Matrix & expectedOutput) 
{
    return Matrix::Zero(output.rows(), output.cols());
}

/**
* This function is a stub. Implement it to return the ReLU output gradient
*/ 
Matrix gradientR(const Matrix &dC, const Matrix &R, const int pooling, int strides)
{
    return Matrix::Zero(R.rows(), R.cols());
}

/**
* This function is a stub. Implement it to return the convolution output gradient 
*/ 
Matrix gradientCONV(const Matrix &dR, const Matrix &CONV)
{
    return Matrix::Zero(CONV.rows(), CONV.cols());
}

/**
* Function to return the kernel gradient
*/
Matrix gradientK(const Matrix &dCONV, const Matrix &X)
{
    Matrix result = convolution(X, dCONV);
    return result;
}

/**
* Usual update rule of gradient descent
*/
void updateK(const Matrix &dK, Matrix &K, const double learningRate)
{
    K = K + (-learningRate * dK);
}

int main(int, char **)
{

    // kernel to be trained
    Matrix K(3, 3);
    K << 
        -1.2, 0.4, -0.1, 
        -0.5, 0.8, 0.9, 
        -0.5, 1.2, 2.1;

    //input matrix
    Matrix X(6, 6);
    X << 
        9, 8, 6, 2, 3, 5,
        8, 9, 7, 3, 4, 4,
        6, 7, 8, 4, 5, 5,
        8, 4, 6, 0, 2, 8,
        5, 6, 5, 5, 3, 2,
        7, 6, 4, 3, 4, 4;

    //expected output 
    Matrix T(2, 2);
    T << 
        0, 10, 
        0, 6;
    
    // Hyperparameters
    const int MAX_EPOCH = 1000;
    const double MIN_COST = 0.001;
    const double learningRate = 0.0001;
    const int poolingSize = 2;

    int epoch = 0;
    double COST;
    Matrix OUTPUT;
    do
    {
        
        /** FORWARD PROPAGATION: **/

        //convolve X by the kernel K
        const Matrix CONV = convolution(X, K);

        //apply activation ReLU
        const Matrix R = ReLU(CONV);

        //obtain the output by using max pooling
        OUTPUT = maxPooling(R, poolingSize, poolingSize);

        //evaluate the quadratic cost for the output
        COST = quadraticCost(OUTPUT, T);

        // don't update the kernel if the cost is suitable
        if(COST <= MIN_COST) break;

        /** BACKWARD PROPAGATION **/

        //find the cost gradient
        const Matrix dC = quadraticCostGradient(OUTPUT, T);

        //find the gradient for max pooling
        const Matrix dR = gradientR(dC, R, poolingSize, poolingSize);

        //find the gradient for ReLU
        const Matrix dCONV = gradientCONV(dR, CONV);

        //finally find the kernel gradient
        Matrix dK = gradientK(dCONV, X);

        //updating the kernel
        updateK(dK, K, learningRate);
        
        epoch++;

    } while(epoch < MAX_EPOCH);

    std::cout << "achieved epoch: " << epoch << "\n\n";
    std::cout << "achieved COST: " << COST << "\n\n";
    std::cout << "K: \n" << K << "\n\n";

    std::cout << "expected OUTPUT: \n" << T << "\n\n";
    std::cout << "OUTPUT: \n" << OUTPUT << "\n\n";

    return 0;
}