#include <iostream>
#include <random>
#include <limits>

#include "csv.h"

#include "cost_functions.hpp"
#include "performance_measurement.hpp"
#include "backpropagation.hpp"

/**
 * Hands-On Neural Network Programming with C++
 * Packt Publishing @ 2019
 * 
 * Chapter 3, example 5
 * 
 * Training a multilayer network with backpropagation
 * 
 * 
**/

std::random_device rd;
std::mt19937 prn(rd());

ann::MultilayerPerceptron initializeNetwork(double initializationRange)
{
    ann::MultilayerPerceptron result;
    Matrix w1 = initializationRange * Matrix::Random(5, 4);
    ann::Layer layer1(std::unique_ptr<ann::ActivationFunction>(new ann::ReLUActivationFunction()), w1, Vector::Zero(5));
    result.add(layer1);

    Matrix w2 = initializationRange * Matrix::Random(4, 5);
    ann::Layer layer2(std::unique_ptr<ann::ActivationFunction>(new ann::ReLUActivationFunction()), w2, Vector::Zero(4));
    result.add(layer2);

    Matrix wOut = initializationRange * Matrix::Random(3, 4);
    ann::Layer outputLayer(std::unique_ptr<ann::ActivationFunction>(new ann::LogisticActivationFunction()), wOut, Vector::Zero(3));
    result.add(outputLayer);
    return result;
}

ann::Dataset loadIrisDataset(const std::string &filepath)
{
    Matrix X = Matrix::Zero(4, 150);
    Matrix T = Matrix::Zero(3, 150);
    io::CSVReader<5> csvReader(filepath);
    csvReader.set_header("sepal_length", "sepal_width", "petal_length", "petal_width", "species");
    double sepal_length, sepal_width, petal_length, petal_width;
    std::string species;
    int colIndex = 0;
    while (csvReader.read_row(sepal_length, sepal_width, petal_length, petal_width, species)){
        X.col(colIndex) << sepal_length, sepal_width, petal_length, petal_width;
        if (species == "Iris-setosa") T(0, colIndex) = 1.0;
        else if (species == "Iris-versicolor") T(1, colIndex) = 1.0;
        else if (species == "Iris-virginica") T(2, colIndex) = 1.0;
        else throw "unknow species";
        colIndex++;
    }
    ann::Dataset result;
    result.X = X;
    result.T = T;
    return result;
}

int main(int, char **)
{
    int epochs = 5000;
    double learningRate = 0.1;
    auto irisDS = loadIrisDataset("../data/iris.csv");
    auto net = initializeNetwork(0.5);
    ann::Backpropagation<ann::QuadraticCostFunction> backpropagation(net, irisDS, learningRate, epochs);
    backpropagation.train();
    double trainingError = ann::mse(net, irisDS);
    std::cout << "The training mse is " << trainingError << "\n";
    return 0;
}