#include <iostream>

#include "csv.h"

#include "activation_functions.hpp"

/**
 * Hands-On Neural Network Programming with C++
 * Packt Publishing @ 2019
 * 
 * Chapter 3, example 3
 * 
 * Training a 4-layer network for the Iris dataset with backpropagation
 * 
 * 
**/

ann::LogisticActivationFunction g;

std::tuple<int, int, double> backpropagation(const Matrix &X, const Matrix &T, double learningRate, int maxEpochs, int minFails);
std::tuple <Matrix, Matrix> loadIrisDataset(const std::string &);

int fails(const Matrix &expected, const Matrix &output)
{
    int result = 0;
    for (int n = 0; n < expected.cols(); n++)
    {
        const Vector &expectedOutput = expected.col(n);
        const Vector &outputCol = output.col(n);
        int predictedClass = std::distance(outputCol.begin(), std::max_element(outputCol.begin(), outputCol.end()));
        int expectedClass = std::distance(expectedOutput.begin(), std::max_element(expectedOutput.begin(), expectedOutput.end()));
        if (predictedClass != expectedClass)
            result++;
    }
 return result;
}

double costFunction(const Matrix &expected, const Matrix &output)
{
    auto cost = output.binaryExpr(expected, [](double y, double t){
        return pow(y - t, 2);
    });
    return cost.sum() / (2*expected.cols());
}


double cost_derivate(const double expected, const double output)
{
    return output - expected;
}

std::tuple<double, int> eval(const Matrix &X, const Matrix &T, const Matrix &w1, const Matrix &w2, const Matrix &w3, const Vector &b1, const Vector &b2, const Vector &b3) {
    auto prod1 = w1 * X;
    auto z1 = prod1.colwise() + b1;
    auto y1 = z1.unaryExpr([](double v) { return g(v); });

    auto prod2 = w2 * y1;
    auto z2 = prod2.colwise() + b2;
    auto y2 = z2.unaryExpr([](double v) { return g(v); });

    auto prod3 = w3 * y2;
    auto z3 = prod3.colwise() + b3;
    auto y3 = z3.unaryExpr([](double v) { return g(v); });

    double cost = costFunction(T, y3);
    int fail = fails(T, y3);
    return std::make_tuple(cost, fail);
}

int main(int, char **)
{
    int maxEpochs = 10000;
    double learningRate = 1;
    double failThreshold = 1;
    auto [X, T] = loadIrisDataset("../data/iris.csv");
    std::cout << "The max number of epochs is " << maxEpochs << "\n";
    std::cout << "Learning rate is " << learningRate << "\n";
    std::cout << "The min number of mispredictions is " << failThreshold << "\n\n";

    auto [epoch, failCount, cost] = backpropagation(X, T, learningRate, maxEpochs, failThreshold);

    std::cout << "\nFinal epoch: " << epoch << "\n";
    std::cout << "Total mispredictions: " << failCount << "\n";
    std::cout << "Final cost: " << cost << "\n";
    return 0;
}

std::tuple<Matrix, Matrix, Matrix, Matrix, Matrix, Matrix> 
forward(const Vector &x, const Matrix &w1, const Matrix &w2, const Matrix &w3, const Vector &b1, const Vector &b2, const Vector &b3)
{
    auto prod1 = w1 * x;
    auto z1 = prod1.colwise() + b1;
    auto y1 = z1.unaryExpr([](double v) { return g(v); });

    auto prod2 = w2 * y1;
    auto z2 = prod2.colwise() + b2;
    auto y2 = z2.unaryExpr([](double v) { return g(v); });

    auto prod3 = w3 * y2;
    auto z3 = prod3.colwise() + b3;
    auto y3 = z3.unaryExpr([](double v) { return g(v); });

    return std::make_tuple(z1, z2, z3, y1, y2, y3);
}

std::tuple<Matrix, Matrix, Matrix, Matrix, Matrix, Matrix> 
backward(const Vector &x, const Vector &t, const Matrix &z1, const Matrix &z2, const Matrix &z3, 
    const Matrix &y1, const Matrix &y2, const Matrix &y3, const Matrix &w2, const Matrix &w3) {

    Matrix sigma = t.binaryExpr(y3, [](double _t, double _y) { return cost_derivate(_t, _y); });
    auto delta3 = sigma.binaryExpr(z3, [](double s, double _z) { return s * g.prime(_z); });
    Matrix dW3_i = delta3 * y2.transpose();
    Vector dB3_i = delta3;

    sigma = w3.transpose() * delta3;
    auto delta2 = sigma.binaryExpr(z2, [](double s, double _z) { return s * g.prime(_z); });
    Matrix dW2_i = delta2 * y1.transpose();
    Vector dB2_i = delta2;

    sigma = w2.transpose() * delta2;
    auto delta1 = sigma.binaryExpr(z1, [](double s, double _z) { return s * g.prime(_z); });
    Matrix dW1_i = delta1 * x.transpose();
    Vector dB1_i = delta1;

    return std::make_tuple(dW1_i, dW2_i, dW3_i, dB1_i, dB2_i, dB3_i);
}

void update(int m, double learningRate,
    Matrix &dW1, Matrix &dW2, Matrix &dW3, Vector &dB1, Vector &dB2, Vector &dB3, 
    Matrix &w1, Matrix &w2, Matrix &w3, Vector &b1, Vector &b2, Vector &b3)
{
    dW1 /= m;
    dB1 /= m;
    dW2 /= m;
    dB2 /= m;
    dW3 /= m;
    dB3 /= m;
    w1 = w1 - learningRate * dW1;
    b1 = b1 - learningRate * dB1;
    w2 = w2 - learningRate * dW2;
    b2 = b2 - learningRate * dB2;
    w3 = w3 - learningRate * dW3;
    b3 = b3 - learningRate * dB3;
}

void initialize(double initializationFactor, Matrix &w1, Matrix &w2, Matrix &w3, Vector &b1, Vector &b2, Vector &b3)
{
    w3 = initializationFactor * Matrix::Random(3, 4); 
    b3 = Vector::Zero(3);
    w2 = initializationFactor * Matrix::Random(4, 5); 
    b2 = Vector::Zero(4);
    w1 = initializationFactor * Matrix::Random(5, 4); 
    b1 = Vector::Zero(5);
}

void zeroInitialization(Matrix &m1, Matrix &m2, Matrix &m3, Vector &v1, Vector &v2, Vector &v3)
{
    m3 = Matrix::Zero(3, 4); 
    v3 = Vector::Zero(3);
    m2 = Matrix::Zero(4, 5); 
    v2 = Vector::Zero(4);
    m1 = Matrix::Zero(5, 4); 
    v1 = Vector::Zero(5);
}

std::tuple<int, int, double> backpropagation(const Matrix &X, const Matrix &T, double learningRate, int maxEpochs, int minFails) {
    Matrix w1, w2, w3; Vector b1, b2, b3;
    initialize(0.5, w1, w2, w3, b1, b2, b3);
    int epoch = 0, m = X.cols();
    auto [cost, failCount] = eval(X, T, w1, w2, w3, b1, b2, b3); 
    while (epoch++ < maxEpochs && failCount > minFails) {
        Matrix dW1, dW2, dW3; Vector dB1, dB2, dB3;
        zeroInitialization(dW1, dW2, dW3, dB1, dB2, dB3);
        for (int i = 0; i < m; ++i) {
            Vector x = X.col(i), t = T.col(i);
            auto [z1, z2, z3, y1, y2, y3] = forward(x, w1, w2, w3, b1, b2, b3);
            auto [dW1_i, dW2_i, dW3_i, dB1_i, dB2_i, dB3_i] = backward(x, t, z1, z2, z3, y1, y2, y3, w2, w3);
            dW1 += dW1_i; dB1 += dB1_i; 
            dW2 += dW2_i; dB2 += dB2_i; 
            dW3 += dW3_i; dB3 += dB3_i;
        }
        update(m, learningRate, dW1, dW2, dW3, dB1, dB2, dB3, w1, w2, w3, b1, b2, b3);
        std::tie(cost, failCount) = eval(X, T, w1, w2, w3, b1, b2, b3); 
    }
    return std::make_tuple(--epoch, failCount, cost);
}

std::tuple<Matrix, Matrix> loadIrisDataset(const std::string &filepath)
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
    return std::make_tuple(X, T);
}