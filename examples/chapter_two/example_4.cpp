#include "mlp_core.hpp"

#include <iostream>

#include "csv.h"

/**
 * Hands-On Neural Network Programming with C++
 * Packt Publishing @ 2019
 * 
 * Chapter 2, example 4
 * 
 * This example shows how to evaluate the network performance by
 * using the mean squared error.
 * 
 * 
**/

Matrix loadDataset(const std::string &filepath);

int main()
{
   // Network definition

   Matrix initialWeights1(4, 2);
   initialWeights1 << 0.069, 0.24, 
                   -0.113, 0.34, 
                   0.069, 0.232, 
                   -2.196, 1.266;
   Vector initialBias1(4);
   initialBias1 << -0.79, -0.895, -0.854, -1.815;
   Layer layer1(LogisticActivationFunction(), initialWeights1, initialBias1);

   Matrix initialWeights2(1, 4);
   initialWeights2 << 0.197, 0.0222, 0.195, -1.294;
   Vector initialBias2(1);
   initialBias2 << 0.138;
   Layer layer2(LogisticActivationFunction(), initialWeights2, initialBias2);

   MultilayerPerceptron net;
   net.add(layer1);
   net.add(layer2);

   // loading dataset and data processing

   auto dataset = loadDataset("../data/circular-example.csv");

   double mse = 0.0;
   for (const auto &instance : dataset.colwise())
   {
       Vector input(2);
       input << instance(0), instance(1);
       Vector expected(1);
       expected << instance(2);
       const Vector output = net.output(input);
       mse += pow(output(0) - expected(0), 2);
   }
   mse /= dataset.size();
   std::cout << "The mean squared error is " << mse << "\n";
   return 0;
}

Matrix loadDataset(const std::string &filepath)
{
    Matrix result(3, 20);
    io::CSVReader<3> csvReader(filepath);
    csvReader.read_header(io::ignore_extra_column, "X0", "X1", "MTYPE");
    double x0, x1, mtype;
    int colIndex = 0;
    while (csvReader.read_row(x0, x1, mtype))
    {
        result.col(colIndex) << x0, x1, mtype;
        colIndex++;
    }
    return result;
}