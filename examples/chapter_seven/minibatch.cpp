#include <iostream>
#include <random>
#include <limits>

#include <mgl2/mgl.h>
#include "csv.h"

#include "cost_functions.hpp"
#include "performance_measurement.hpp"
#include "backpropagation.hpp"

//std::random_device rd;
std::mt19937 prn(4);

ann::MultilayerPerceptron initializeNetwork(double initializationRange)
{
    ann::MultilayerPerceptron result;
    Matrix w1 = initializationRange * Matrix::Random(5, 4);
    ann::Layer layer1(std::unique_ptr<ann::ActivationFunction>(new ann::LogisticActivationFunction()), w1, Vector::Zero(5));
    result.add(layer1);

    Matrix w2 = initializationRange * Matrix::Random(4, 5);
    ann::Layer layer2(std::unique_ptr<ann::ActivationFunction>(new ann::LogisticActivationFunction()), w2, Vector::Zero(4));
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

void drawMSE(Matrix &mseData)
{
    mglData model1, model2, model3;
    model1.Create(mseData.cols(), 1);
    model2.Create(mseData.cols(), 1);
    model3.Create(mseData.cols(), 1);

    double maxMSE = mseData.maxCoeff();

    for (unsigned i = 0; i < mseData.cols(); i++)
    {
        model1.a[i] = mseData(0, i);
        model2.a[i] = mseData(1, i);
        model3.a[i] = mseData(2, i);
    }

    mglGraph gr;

    gr.SetSize(800, 480);
    gr.Title("Training MSE on Iris Dataset");

    gr.SetRanges(0, mseData.cols(), 0, 1.1*maxMSE);	
    gr.Label('x', "epoch x 100", 0);
    gr.Axis();	
    gr.Plot(model1, "|b");
    gr.Plot(model2, "jq");
    gr.Plot(model3, "=H");

    gr.AddLegend("Deterministic GD", "|b");
    gr.AddLegend("Minibatch 32", "jq");
    gr.AddLegend("SGD", "=H");
    gr.Legend(1, 1.1);

    gr.WriteFrame("chapter7.svg");

}

int main(int argc, char **argv)
{
    int epochs = 10000;
    double learningRate = 0.1;
    if (argc > 1)
    {
        int epochsParam = std::atoi(argv[1]);
        if (epochsParam > 0)
            epochs = epochsParam;
    }
    if (argc > 2)
    {
        double learningRateParam = std::stod(argv[2]);
        if (learningRateParam > 0)
            learningRate = learningRateParam;
    }

    auto fullData = loadIrisDataset("../data/iris.csv");
    ann::shuffleDataset(fullData, prn);

    std::cout << "Training with deterministic gradient descent\n";

    //model 1, regular gradient descent
    auto data1 = fullData;
    auto net1 = initializeNetwork(0.5);
    ann::Backpropagation<ann::QuadraticCostFunction> deterministicGD(net1, data1, learningRate, epochs);
    auto trainingError1 = deterministicGD.train();

    std::cout << "Training with minibatch size 32\n";

    //model 2, minibatch size 32
    auto data2 = fullData;
    auto net2 = initializeNetwork(0.5);
    int minibatchSize = 32;
    ann::Backpropagation<ann::QuadraticCostFunction> minibatchGD(net2, data2, learningRate, epochs, minibatchSize);
    auto trainingError2 = minibatchGD.train();

    std::cout << "Training model stochastic gradient descent\n";

    //model 3, with SGD
    auto data3 = fullData;
    auto net3 = initializeNetwork(0.5);
    minibatchSize = 1;
    ann::Backpropagation<ann::QuadraticCostFunction> 
        stochasticGD(net3, data3, learningRate, epochs, minibatchSize);
    auto trainingError3 = stochasticGD.train();

    auto mseData = Matrix(3, trainingError1.cols());
    mseData.row(0) = trainingError1.row(0);
    mseData.row(1) = trainingError2.row(0);
    mseData.row(2) = trainingError3.row(0);

    drawMSE(mseData);

    return 0;
}