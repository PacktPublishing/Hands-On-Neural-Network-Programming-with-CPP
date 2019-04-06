#include "activation_functions.hpp"

#include <iostream>

/**
 * Hands-On Neural Network Programming with C++
 * Packt Publishing @ 2019
 * 
 * Chapter 2, example 3
 * 
 * This example shows how to use the softmax activation function
 * 
 * 
**/

int main()
{

   Matrix input(3, 1);
   input << 5.0, 0.0, 2.0;

   SoftmaxActivationFunction softmax;
   auto output = softmax(input);

   std::cout << output << "\n";

   return 0;
}