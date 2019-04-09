#include <iostream>
#include <numeric>
#include <random>

#include "activation_functions.hpp"

/**
 * Hands-On Neural Network Programming with C++
 * Packt Publishing @ 2019
 * 
 * Chapter 3, example 2
 * 
 * Training a small network with gradient descent
 * 
 * 
**/

std::tuple<double, double, double, double> gradientDescent(Matrix &X, Matrix &T, double learningRate, int maxEpochs, double minCost);
std::random_device rd;
std::mt19937 prn(rd());
std::uniform_real_distribution<> uniformDist(-1.0, 1.0);

ann::LogisticActivationFunction g;

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

int main(int, char **)
{
    int epochs = 500;
    double learningRate = 5;
    double costThreshold = 1e-2;
    auto [X, T] = makeSyntheticDataset();
    std::cout << "The max number of epochs is " << epochs << "\n";
    std::cout << "Learning rate is " << learningRate << "\n";
    std::cout << "The target cost is " << costThreshold << "\n\n";
    auto [epoch, cost, w, b] = gradientDescent(X, T, learningRate, epochs, costThreshold);
    std::cout << "Reached epoch: " << epoch << "\n";
    std::cout << "Final cost: " << cost << "\n";
    std::cout << "Final Weights: " << w << "\n";
    std::cout << "Final bias: " << b << "\n";
    return 0;
}

double dCdz(double w, double b, double x, double t)
{
    double z = x * w + b;
    double y = g(z);
    double result = (y - t) * y * (1 - y);
    return result;
}

std::tuple<double, double, double, double>  gradientDescent(Matrix &X, Matrix &T, const double learningRate, const int maxEpochs, const double minCost) {
    double b = 0.0, w = 0.5 * uniformDist(prn);
    double cost = quadraticCost(X, T, w, b);
    int epoch = 0, m = X.cols();
    while (epoch++ < maxEpochs && cost > minCost) {
        double dW = 0.0, dB = 0.0;
        for (int i = 0; i < m; ++i) {
            double x = X(0, i), t = T(0, i);
            double dz = dCdz(w, b, x, t);
            dW += dz * x;
            dB += dz;
        }
        dW /= m;     dB /= m;
        w = w - learningRate * dW;
        b = b - learningRate * dB;
        cost = quadraticCost(X, T, w, b);
    }
    return std::make_tuple(--epoch, cost, w, b);
}