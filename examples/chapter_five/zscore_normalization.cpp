#include <iostream>
#include <limits>

#include <mgl2/mgl.h>
#include "csv.h"

#include "performance_measurement.hpp"

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

void drawCorrelationIris(mglGraph &gr, Matrix &features, int attribute0, int attribute1)
{
    mglData dataX, dataY;
    dataX.Create(150, 1);
    dataY.Create(150, 1);

    for (unsigned i = 0; i < 150; i++)
    {
        dataX.a[i] = features(attribute0, i);
        dataY.a[i] = features(attribute1, i);
    }

    double minXValue = features.row(attribute0).minCoeff();
    double maxXValue = features.row(attribute0).maxCoeff();
    double minYValue = features.row(attribute1).minCoeff();
    double maxYValue = features.row(attribute1).maxCoeff();

    std::cout << "minXValue:\t" << minXValue << "\n";
    std::cout << "maxXValue:\t" << maxXValue << "\n";
    std::cout << "minYValue:\t" << minYValue << "\n";
    std::cout << "maxYValue:\t" << maxYValue << "\n";
    std::cout << "==================\n";

    double padX = abs(maxXValue - minXValue) / 2.0;
    double padY = abs(maxYValue - minYValue) / 2.0;
    double chartXlow = minXValue - padX;
    double chartXHigh = maxXValue + padX;
    double chartYlow = minYValue - padY;
    double chartYHigh = maxYValue + padY;
    double hRange = chartXHigh - chartXlow;
    double vRange = chartYHigh - chartYlow;

    int width = 800;
    gr.SetSize(width, width * vRange / hRange);

    gr.SetOrigin(0, 0, 0);
    gr.SetRanges(chartXlow, chartXHigh, chartYlow, chartYHigh);	
    gr.Axis();	
    gr.Box();
    gr.Plot(dataX, dataY, " xr");

}

int main(int, char **)
{

    auto iris = loadIrisDataset("../data/iris.csv");

    mglGraph gr;

    int attribute0 = 2;
    int attribute1 = 3;

    //std::cout << "original data\n";
    //drawCorrelationIris(gr, iris.X, attribute0, attribute1);

    Vector means = iris.X.rowwise().mean();

    Matrix newX = iris.X.colwise() - means;

    //std::cout << "\nremoving the mean\n";
    //drawCorrelationIris(gr, newX, attribute0, attribute1);

    Matrix power = newX.array().pow(2);

    int n = newX.cols() - 1;

    Vector div = power.rowwise().sum() / n;

    Vector stddev = div.array().sqrt();

    Matrix normalized = newX.array().colwise() / stddev.array();

    std::cout << "\nscaling up\n";
    drawCorrelationIris(gr, normalized, attribute0, attribute1);

    gr.WriteFrame("zscore.svg");

    return 0;
}