#include <iostream>
#include <math.h>

/**
 * Hands-On Neural Network Programming with C++
 * Packt Publishing @ 2019
 * 
 * Chapter 1, example 2
 * 
 * Example of binary classification using the network outcome
 * 
 * This program is an simply example on how to use the network output
 * to classify instances in two different categories
 * 
**/
double g(double z);
double naiveFeedforwardPropagation(double input[]);
char classification(double score);

int main(int argc, char* argv[]) {
    if(argc < 3) {
        std::cerr << "Please provide the command line arguments x0 and x1.\n";
        return -1;
    }
    double x0 = std::stod(argv[1]);
    double x1 = std::stod(argv[2]);
    double input[] = { x0, x1 };
    double response = naiveFeedforwardPropagation(input);
    std::cout << "input: {" << input[0] << ", " << input[1] << "}\n";
    std::cout << "network output: " << response << "\n";
    std::cout << "classified as: " << classification(response) << "\n";
    return 0;
}

double g(double z) {
   if(z >= 45) return 1.0;
   else if(z <= -45) return 0.0;
   return 1.0 / (1.0 + exp(-z));
}

double naiveFeedforwardPropagation(double input[]) {
    double bias = -2.0;
    double weights[] = { -3.1, 7.1 };

    double z = bias;
    z += input[0] * weights[0];
    z += input[1] * weights[1];
    return g(z);
}

char classification(double score) {
    char result = 'B';
    if(score > .5) result = 'A';
    return result;
}