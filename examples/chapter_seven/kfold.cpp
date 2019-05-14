#include <iostream>
#include <random>
#include <limits>

#include "csv.h"

#include "cost_functions.hpp"
#include "performance_measurement.hpp"
#include "backpropagation.hpp"

std::random_device rd;
std::mt19937 prn(rd());

ann::Dataset loadIrisDataset(const std::string &filepath)
{
    Matrix X = Matrix::Zero(4, 150);
    Matrix T = Matrix::Zero(3, 150);
    io::CSVReader<5> csvReader(filepath);
    csvReader.set_header("sepal_length", "sepal_width", "petal_length", "petal_width", "species");
    double sepal_length, sepal_width, petal_length, petal_width;
    std::string species;
    int colIndex = 0;
    while (csvReader.read_row(sepal_length, sepal_width, petal_length, petal_width, species))
    {
        X.col(colIndex) << sepal_length, sepal_width, petal_length, petal_width;
        if (species == "Iris-setosa")
            T(0, colIndex) = 1.0;
        else if (species == "Iris-versicolor")
            T(1, colIndex) = 1.0;
        else if (species == "Iris-virginica")
            T(2, colIndex) = 1.0;
        else
            throw "unknow species";
        colIndex++;
    }
    ann::Dataset result;
    result.X = X;
    result.T = T;
    return result;
}

ann::MultilayerPerceptron initializeNetwork(const int numberOfHiddenLayers = 1, const int numberOfNeuronsInHiddenLayer = 10)
{
    ann::MultilayerPerceptron result;
    double initializationRange = 0.05;

    int numberOfInputNeurons = 4;
    for (int i = 0; i < numberOfHiddenLayers; i++)
    {
        Matrix w = initializationRange * Matrix::Random(numberOfNeuronsInHiddenLayer, numberOfInputNeurons);
        ann::Layer layer(std::unique_ptr<ann::ActivationFunction>(new ann::LogisticActivationFunction), w, Vector::Zero(numberOfNeuronsInHiddenLayer));
        result.add(layer);
        numberOfInputNeurons = numberOfNeuronsInHiddenLayer;
    }

    Matrix wOut = initializationRange * Matrix::Random(3, numberOfInputNeurons);
    ann::Layer outputLayer(std::unique_ptr<ann::ActivationFunction>(new ann::LogisticActivationFunction()), wOut, Vector::Zero(3));
    result.add(outputLayer);

    return result;
}

int main()
{
    double learnRate = 1.0;
    int epochs = 8'000;
    auto dataset = loadIrisDataset("../data/iris.csv");
    shuffleDataset(dataset, prn);
    int dataSize = dataset.size();
    int k = 4;
    int foldSize = dataSize / k;
    int remains = dataSize % k;
    double finalMSE = .0;
    int foldEnd = 0;
    for (int i = 0; i < k; ++i)
    {
        int foldBegin = foldEnd;
        foldEnd = foldBegin + foldSize;
        if (remains > 0)
        {
            foldEnd++;
            remains--;
        }
        auto trainDS = dataset;
        auto validDS = trainDS.remove(foldBegin, foldEnd);
        auto net = initializeNetwork();
        ann::Backpropagation<ann::QuadraticCostFunction> bp(net, trainDS, learnRate, epochs);
        bp.train();
        finalMSE += mse(net, validDS);
    }
    std::cout << "The estimated generalization MSE is\t" << finalMSE / k << "\n";
    return 0;
}