#include <iostream>
#include <iomanip>
#include <random>
#include <limits>

#include "csv.h"

#include "cost_functions.hpp"
#include "performance_measurement.hpp"
#include "backpropagation.hpp"

ann::MultilayerPerceptron initializeNetwork(double initializationRange)
{
    ann::MultilayerPerceptron result;
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
    std::cout << std::setprecision(16);
    int epochs = 100;
    double learningRate = 0.5;
    auto irisDS = loadIrisDataset("../data/iris.csv");
    auto net = initializeNetwork(0.5);
    ann::Backpropagation<ann::QuadraticCostFunction> backpropagation(net, irisDS, learningRate, epochs);
    backpropagation.train();
    auto metrics = ann::evaluate(net, irisDS);

    std::cout << std::setprecision(3);

    std::cout << "Confusion Matrix:\n";
    std::cout << metrics.confusionMatrix << "\n\n";
    for(int k = 0; k < 3; ++k)
        std::cout << "precision class " << k << ": " << metrics.precision(k) << "\n";
    std::cout << "\n";
    for(int k = 0; k < 3; ++k)
        std::cout << "recall class " << k << ": " << metrics.recall(k) << "\n";
    std::cout << "\n";
    for(int k = 0; k < 3; ++k)
        std::cout << "specificity class " << k << ": " << metrics.specificity(k) << "\n";
    std::cout << "\n";
    for(int k = 0; k < 3; ++k)
        std::cout << "accuracy class " << k << ": " << metrics.accuracy(k) << "\n";
    std::cout << "\n";
    for(int k = 0; k < 3; ++k)
        std::cout << "f1Score class " << k << ": " << metrics.f1Score(k) << "\n";
    std::cout << "\n";

    return 0;
}