#include <iostream>
#include <numeric>
#include <random>

#include <mgl2/mgl.h>

#include "matrix_definitions.hpp"

/**
 * Hands-On Neural Network Programming with C++
 * Packt Publishing @ 2019
 * 
 * Chapter 3, example 2
 * 
 * Find gradient vectors for the weight parameter
 * 
 * 
**/

std::random_device rd;
std::mt19937 prn(rd());
std::uniform_real_distribution<> dis(0.0, 1.0);

int gradTick = 3;
double wStep = 0.1;
double bStep = wStep;

double limite = 8.0;
double wMin = -limite;
double wMax = limite;
double bMin = -limite;
double bMax = limite;

const int n = static_cast<int>((wMax - wMin) / wStep) + 1;
const int m = static_cast<int>((bMax - bMin) / bStep) + 1;

long slice = static_cast<long>((-2.5 - bMin) * n / (bMax - bMin));

double g(const double z)
{
    double result;
    if (z >= 45.0) result = 1;
    else if (z <= -45.0) result = 0;
    else result = 1.0 / (1.0 + exp(-z));
    return result;
}

double derivate(double x, double y, double w, double b)
{
    double z = x * w + b;
    double a = g(z);
    double result = (a - y) * a * (1. - a);
    result = result * x;
    return result;
}

double derivate(const Matrix &xt, double w, double b)
{
    auto XTcolumnwise = xt.colwise();
    double sum = 0.0;
    std::for_each(XTcolumnwise.begin(), XTcolumnwise.end(), [w, b, &sum](const auto &column) {
        double x = column(0);
        double y = column(1);
        sum += derivate(x, y, w, b);
    });
    return sum / xt.cols();
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

std::tuple<mglData, mglData, mglData, mglData, mglData, mglData> convertChartData(const Matrix &data, const Matrix &gradients)
{
    mglData _3Dresult, _2Dresult, gradFromX, gradFromY, gradX, gradY;
    int i, j, n = data.rows(), m = data.cols();
    _3Dresult.Create(n, m);
    _2Dresult.Create(n);

    gradFromX.Create(gradients.rows(), 2);
    gradFromY.Create(gradients.rows(), 2);
    gradX.Create(gradients.rows(), 2);
    gradY.Create(gradients.rows(), 2);
    for (i = 0; i < n; i++)
    {
        if (i % gradTick == 0)
        {
            int gradIndex = i / gradTick;
            if (gradIndex < gradients.rows())
            {
                gradFromX[gradIndex] = gradients(gradIndex, 0);
                gradFromY[gradIndex] = gradients(gradIndex, 1);
                gradX[gradIndex] = gradTick * wStep;
                gradY[gradIndex] = gradX[gradIndex] * gradients(gradIndex, 2);
            }
        }
        for (j = 0; j < m; j++)
        {
            int _3Dindex = i + n * j;
            _3Dresult.a[_3Dindex] = data(i, j);
            if (j == slice)
                _2Dresult.a[i] = data(i, j);
        }
    }
    return std::make_tuple(_3Dresult, _2Dresult, gradFromX, gradFromY, gradX, gradY);
}

void generateChartDense(mglGraph &gr, const mglData &a)
{

    gr.Label('x', "W", 0);
    gr.Label('y', "b", 0);

    gr.SetRanges(wMin, wMax, bMin, bMax);
    gr.Axis();
    gr.Box();
    gr.Dens(a, "#", "meshnum 7.5");
    gr.Colorbar();
}

void generateChartSurface(mglGraph &gr, const mglData &a, double ang1, double ang2, double zMin, double zMax)
{

    gr.Rotate(ang1, ang2);
    gr.SetRanges(wMin, wMax, bMin, bMax, zMin, zMax);
    gr.Axis();
    gr.Surf(a, "#", "meshnum 7.5");
    gr.Grid();
    gr.Label('x', "w", 0);
    gr.Label('y', "b", 0);

    long i = slice;
    double b = ((bMax - bMin) * i / m) + bMin;
    mglData x(n);
    mglData y(n);
    x.Fill(wMin, wMax);
    y.Fill(b, b);
    mglData z(a.SubData(-1,i));
    gr.Plot(x, y, z, "2r");
}

void generateChartW(const char *xLabel, mglGraph &gr, const mglData &data, double zMin, double zMax, const mglData &gradFromX, const mglData &gradFromY, const mglData &gradX, const mglData &gradY)
{

    gr.Label('x', xLabel, 0);
    gr.Light(false);
    gr.Alpha(false);

    gr.SetRanges(bMin, bMax, zMin, zMax);
    gr.SetOrigin(bMin, zMin);
    gr.Grid();
    gr.Axis();
    gr.Plot(data, "r-2");
    gr.SetArrowSize(1.0);
    gr.Traj(gradFromX, gradFromY, gradX, gradY, "b4", "value 0.5");
}

void generateChart(const Matrix &data, const Matrix &gradients, double zMin, double zMax)
{
    mglGraph gr;
    gr.SetSize(1000, 480);
    gr.Title("Cost x weight x bias");
    auto [_3DData, _2DData, gradFromX, gradFromY, gradX, gradY] = convertChartData(data, gradients);

    gr.SubPlot(2, 1, 0);

    generateChartSurface(gr, _3DData, 75, 20, zMin, zMax);

    gr.SubPlot(2, 1, 1);

    generateChartW("w", gr, _2DData, 0, 0.3, gradFromX, gradFromY, gradX, gradY);

    gr.WriteFrame("gradient_weight.svg");
}

int main(int, char **)
{

    Matrix xt(2, 20);
    xt.block(1, 0, 1, 20) << 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 0;

    auto xtColwise = xt.colwise();

    std::transform(xtColwise.begin(), xtColwise.end(), xtColwise.begin(), [](const auto &column) {
        double t = column(1);
        double x = 2.0 + 3 * t + dis(prn);
        Vector result(2);
        result(0) = x;
        result(1) = t;
        return result;
    });

    Matrix X = xt.row(0);
    Matrix T = xt.row(1);

    Matrix data(n, m);

    int gradientsNumber = data.cols() / gradTick;

    Matrix gradients(gradientsNumber, 3);

    double zMin = std::numeric_limits<double>::max();
    double zMax = std::numeric_limits<double>::min();

    for (int i = 0; i < n; ++i)
    {
        for (int j = 0; j < m; ++j)
        {
            double w = wMin + i * wStep;
            double b = bMin + j * bStep;

            double cost = quadraticCost(X, T, w, b);

            if ((j == slice) && (i % gradTick == 0))
            {
                int gradIndex = i / gradTick;

                if (gradIndex < gradientsNumber)
                {
                    double parcialDerivate = derivate(xt, w, b);
                    gradients(gradIndex, 0) = w;
                    gradients(gradIndex, 1) = cost;
                    gradients(gradIndex, 2) = parcialDerivate;
                }
            }

            data(i, j) = cost;

            if (cost > zMax)
            {
                zMax = cost;
            }
            if (cost < zMin)
            {
                zMin = cost;
            }
        }
    }

    generateChart(data, gradients, zMin, zMax);

    return 0;
}