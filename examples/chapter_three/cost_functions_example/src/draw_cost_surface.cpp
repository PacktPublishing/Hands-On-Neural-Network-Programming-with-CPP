#include <iostream>
#include <numeric>
#include <random>

#include <mgl2/mgl.h>

#include "matrix_definitions.hpp"

/**
 * Hands-On Neural Network Programming with C++
 * Packt Publishing @ 2019
 * 
 * Chapter 3, example 1
 * 
 * Generating the cost surface
 * 
 * 
**/

double g(const double z)
{
    double result;
    if (z >= 45.0) result = 1;
    else if (z <= -45.0) result = 0;
    else result = 1.0 / (1.0 + exp(-z));
    return result;
}

double quadraticCost(const Matrix & X, const Matrix & T, double w, double b)
{
    auto output = X.unaryExpr([w,b](double x){
        return g(x * w + b);
    });
    auto cost = output.binaryExpr(T, [](double y, double t){
        return pow(y - t, 2);
    });
    return cost.sum() / (2.0 * X.cols());
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

void generateChartSurface(mglGraph &gr, const mglData &a, double ang1, double ang2, 
    double xMin, double xMax, double yMin, double yMax, double zMin, double zMax)
{
    gr.Rotate(ang1, ang2);
    gr.SetRanges(xMin, xMax, yMin, yMax, zMin, zMax);
    gr.Axis();
    gr.Surf(a, "#", "meshnum 7.5");
    gr.Grid();
    gr.Label('x', "w", 0);
    gr.Label('y', "b", 0);
}

void generateChartDense(mglGraph &gr, const mglData &a, double xMin, double xMax, 
    double yMin, double yMax)
{

    gr.Label('x', "w", 0);
    gr.Label('y', "b", 0);

    gr.Light(false);
    gr.Alpha(false);
    gr.SetRanges(xMin, xMax, yMin, yMax);
    gr.Axis();
    gr.Box();
    gr.Dens(a);
    gr.Grid();
    gr.Colorbar();
}

void generateChart(const Matrix &data)
{
    mglGraph gr;
    gr.SetSize(1440, 480);
    gr.Title("Cost x Weights & bias");
    auto chartData = convertChartData(data);

    gr.SubPlot(2, 1, 0);
    generateChartSurface(gr, chartData, 60, 40, -8.0, 8.0, -8.0, 8.0, data.minCoeff(), data.maxCoeff());

    gr.SubPlot(2, 1, 1);

    generateChartDense(gr, chartData, -8.0, 8.0, -8.0, 8.0);
    gr.WriteFrame("cost_x_w_b.svg");
}

int main(int, char **)
{
    auto [X, T] = makeSyntheticDataset();
    Matrix surface = generateCostSurface(X, T, -8, 8, -8, 8, 0.1);
    generateChart(surface);
    return 0;
}