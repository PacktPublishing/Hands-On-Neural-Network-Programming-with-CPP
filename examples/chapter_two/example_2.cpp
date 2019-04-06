#include "mlp_core.hpp"

#include <iostream>

/**
 * Hands-On Neural Network Programming with C++
 * Packt Publishing @ 2019
 * 
 * Chapter 2, example 2
 * 
 * Multilayer perceptron example
 * 
 * 
**/

int main()
{
    Matrix initialWeights1(2, 3);
    initialWeights1 << .1, -.4, .1, -.3, .3, -.1;
    Vector initialBias1(2);
    initialBias1 << .3, .2;
    Layer layer1(LogisticActivationFunction(), initialWeights1, initialBias1);

    Matrix initialWeights2(1, 2);
    initialWeights2 << 20.0, -21.0;
    Vector initialBias2(1);
    initialBias2 << -.4;
    Layer layer2(LogisticActivationFunction(), initialWeights2, initialBias2);

    MultilayerPerceptron net;
    net.add(layer1);
    net.add(layer2);

    Vector input(3);
    input << 1.0, .5, .8;
    auto output = net.output(input);
    std::cout << "The network output is " << output(0) << "\n";

    return 0;
}