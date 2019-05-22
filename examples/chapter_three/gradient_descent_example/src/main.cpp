#include <iostream>
#include <numeric>
#include <random>

#include "matrix_definitions.hpp"

/**
 * Hands-On Neural Network Programming with C++
 * Packt Publishing @ 2019
 * 
 * Chapter 3, example 4
 * 
 * Training a small network with gradient descent
 * 
 * 
**/

std::random_device rd;
std::mt19937 prn(rd());
std::uniform_real_distribution<> uniformDist(-1.0, 1.0);

double g(const double z)
{
    double result;
    if (z >= 45.0) result = 1;
    else if (z <= -45.0) result = 0;
    else result = 1.0 / (1.0 + exp(-z));
    return result;
}

std::tuple<Matrix, Matrix> makeSyntheticDataset()
{
    Matrix T(1, 20);
    T << 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 0;
    std::uniform_real_distribution<> randUniform(0.0, 1.0);
    Matrix X = T.unaryExpr([&randUniform](double t){
        return 2.0 + 3 * t + randUniform(prn);
    });
    return std::make_tuple(X, T);
}

double quadraticCost(const Matrix & X, const Matrix & T, double w, double b)
{
    auto output = X.unaryExpr([w,b](double x){
        return g(x * w + b);
    });
    auto cost = output.binaryExpr(T, [](double y, double t){
        return pow(y - t, 2);
    });
    return cost.sum() / (2*X.cols());
}

double calc_dZ(double w, double b, double x, double t)
{
    double z = x * w + b;
    double y = g(z);
    double result = (y - t) * y * (1 - y);
    return result;
}

std::tuple<double, double, double, double>
gradientDescent(Matrix &X, Matrix &T, double learningRate, int maxEpochs, double minCost) {
    double b = 0.0;
    double w = 0.5 * uniformDist(prn);
    double cost = quadraticCost(X, T, w, b);
    int epoch = 0;
    while (epoch < maxEpochs && cost > minCost) {

        auto dZ = X.binaryExpr(T, [&w, &b](double x, double t){
            return calc_dZ(w, b, x, t);
        });
        double dW = dZ.cwiseProduct(X).mean();
        double dB = dZ.mean();

        w = w - learningRate * dW;   
        b = b - learningRate * dB;
        cost = quadraticCost(X, T, w, b);
        epoch++;
    }
    return std::make_tuple(epoch, cost, w, b);
}

int main(int, char **)
{
    int epochs = 500;
    double learningRate = 5;
    double costThreshold = 1e-2;
    auto [X, T] = makeSyntheticDataset();
    std::cout << "Max number of epochs: " << epochs << " | ";
    std::cout << "Learning rate: " << learningRate << " | ";
    std::cout << "Target cost: " << costThreshold << "\n";

    auto [epoch, cost, w, b] = gradientDescent(X, T, learningRate, epochs, costThreshold);
    std::cout << "Reached epoch: " << epoch << " | ";
    std::cout << "Final cost: " << cost << "\n";
    std::cout << "Final Weights: " << w << " | ";
    std::cout << "Final bias: " << b << "\n";
    return 0;
}