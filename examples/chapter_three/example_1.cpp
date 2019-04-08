#include <iostream>
#include <numeric>
#include <random>

#include <Eigen/Core>

using Matrix = Eigen::MatrixXd;
using Vector = Eigen::VectorXd;

#include <mgl2/mgl.h>

#include "activation_functions.hpp"

void generateChart(const Matrix &data);

ann::LogisticActivationFunction g;

std::tuple<Matrix, Matrix> loadDataset();

double quadraticCost(const Matrix & X, const Matrix & T, double w, double b)
{
    auto output = X.unaryExpr([w,b](double x){
        return g.evaluate(x * w + b);
    });
    auto cost = output.binaryExpr(T, [](double y, double t){
        return pow(y - t, 2);
    });
    return cost.sum() / X.cols();
}

std::tuple<Matrix, Matrix> makeSyntheticDataset()
{
    Matrix T(1, 20);
    T << 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 0;

    std::random_device rd;
    std::mt19937 prn(rd());
    std::uniform_real_distribution<> randUniform(0.0, 1.0);

    Matrix X = T.unaryExpr([&randUniform, &prn](double t){
        return 2.0 + 3 * t + randUniform(prn);
    });
    return std::make_tuple(X, T);
}

Matrix generateCostSurface(const Matrix &X, const Matrix &T, double wMin, double wMax, double bMin, double bMax, double step) 
{
    int n = static_cast<int>((wMax - wMin)/step);
    int m = static_cast<int>((bMax - bMin)/step);
    Matrix result(n, m);
    for (int i = 0; i < n; ++i)
    {
        double w = wMin + i * step;
        for (int j = 0; j < m; ++j)
        {
            double b = bMin + j * step;
            double cost = quadraticCost(X, T, w, b);
            result(i, j) = cost;
        }
    }
    return result;
}

int main(int, char **)
{
    auto [X, T] = makeSyntheticDataset();
    Matrix surface = generateCostSurface(X, T, -8, 8, -8, 8, 0.1);
    generateChart(surface);
    return 0;
}

mglData convertChartData(const Matrix &data)
{
    mglData result;
    int n = data.rows();
    result.Create(n, n);

    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j++)
        {
            int index = i + n * j;
            result.a[index] = data(i, j);
        }
    }
    return result;
}

void generateChartSurface(mglGraph &gr, const mglData &a, double ang1, double ang2, double zMin, double zMax)
{
    gr.Rotate(ang1, ang2);
    gr.SetRanges(-8.0, 8.0, -8.0, 8.0, zMin, zMax);
    gr.Axis();
    gr.Surf(a, "#", "meshnum 7.5");
    gr.Grid();
    gr.Label('x', "W00", 0);
    gr.Label('y', "b", 0);
}

void generateChart(const Matrix &data)
{
    mglGraph gr;
    gr.Title("Cost x Weights & bias");
    auto chartData = convertChartData(data);
    generateChartSurface(gr, chartData, 60, 60, data.minCoeff(), data.maxCoeff());
    gr.WriteFrame("MLP_w_b.png");
}