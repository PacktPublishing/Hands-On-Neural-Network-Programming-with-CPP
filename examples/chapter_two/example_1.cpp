#include "mlp_core.hpp"
#include "dataset.hpp"

#include <iostream>
#include <iomanip>
#include <numeric>

/**
 * Hands-On Neural Network Programming with C++
 * Packt Publishing @ 2019
 * 
 * Chapter 2, example 1
 * 
 * Example of usage of logistic sigmoid and tanh activation function
 * 
 * 
**/


int main()
{
    LogisticActivationFunction logisticSigmoid;
    double z = 0.0;
    double y = logisticSigmoid(z);
    std::cout << "g(" << z << ") is " << y << "\n";

    std::vector<double> myVector = {-50., -1.0, 0.0, 1.0, 50.0};
    std::vector<double> results(myVector.size(), 0.0);

    TanhActivationFunction tanhActivation;
    std::transform(myVector.begin(), myVector.end(), results.begin(), tanhActivation);

    auto print = [](const double& n) { std::cout << n << ", " ; };
    std::for_each(results.begin(), results.end(), print);
    std::cout << "\n";

    return 0;
}