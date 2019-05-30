#include <iostream>
#include <math.h>

/**
 * Hands-On Neural Network Programming with C++
 * Packt Publishing @ 2019
 * 
 * Chapter 1, example 1
 * 
 * Proof-of-Concept of the feed-forward propagation
 * 
 * This program is a motivating example of the computations found in the feed-forward propagation 
 * routine of multi layer perceptron networks. This code also exposes the pitfalls to be avoided 
 * in the forthcoming implementations using functional-style constructs.
 * 
**/
double g(double z);
double neuronOutput(double w [], double b, double previousNeurons [], unsigned numberOfPreviousNeurons);
double naiveFeedforwardPropagation(double input []);

int main() 
{
    double input [] = {1.0, 0.5, 0.8};
    double response = naiveFeedforwardPropagation(input);
    std::cout << "The response for the input {";
    std::cout << input[0] << ", " << input[1] << ", " << input[2];
    std::cout << "} is " << response << "!\n";
    return 0;
}

double g(double z) {
   if(z >= 45) return 1.0;
   else if(z <= -45) return 0.0;
   return 1.0 / (1.0 + exp(-z));
}

double neuronOutput(double w [], double b, double previousNeurons [], int numberOfPreviousNeurons) 
{
    double z = b;
    for(int i = 0; i < numberOfPreviousNeurons; i++) {
        z += w[i] * previousNeurons[i];
    }
    return g(z);
}

double naiveFeedforwardPropagation(double input []) 
{
    // network configuration
    double neuron_0_layer_1_weights [] = {.1, -.4, .1}; double bias_0_layer_1 = .3;
    double neuron_1_layer_1_weights [] = {-.3, .3, -.1}; double bias_1_layer_1 = .2;
    double neuron_0_layer_2_weights [] = {20.0, -21.0}; double bias_0_layer_2 = -.4;

    //output of the neuron n0 of the hidden layer
    double neuron_0_layer_1 = neuronOutput(neuron_0_layer_1_weights, bias_0_layer_1, input, 3); 
    //output of the neuron n0 of the first layer
    double neuron_1_layer_1 = neuronOutput(neuron_1_layer_1_weights, bias_1_layer_1, input, 3);

    double hiddenNeuronsOutputs [] = {neuron_0_layer_1, neuron_1_layer_1};
    double neuron_0_layer_2 = neuronOutput(neuron_0_layer_2_weights, bias_0_layer_2, hiddenNeuronsOutputs, 2);
    return neuron_0_layer_2;
}