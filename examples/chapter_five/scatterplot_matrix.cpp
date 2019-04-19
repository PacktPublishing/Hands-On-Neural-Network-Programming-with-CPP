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

void drawCorrelationIris(mglGraph &gr, ann::Dataset &dataset, int attribute0, int attribute1, std::vector<std::string> &attributeNames)
{
    mglData setosaX, setosaY, versicolorX, versicolorY, virginicaX, virginicaY;
    setosaX.Create(50, 1);
    setosaY.Create(50, 1);
    versicolorX.Create(50, 1);
    versicolorY.Create(50, 1);
    virginicaX.Create(50, 1);
    virginicaY.Create(50, 1);

    Matrix &features = dataset.X;

    for (unsigned i = 0; i < 50; i++)
    {
        setosaX.a[i] = features(attribute0, i);
        setosaY.a[i] = features(attribute1, i);
    }

    for (unsigned i = 50; i < 100; i++)
    {
        versicolorX.a[i - 50] = features(attribute0, i);
        versicolorY.a[i - 50] = features(attribute1, i);
    }

    for (unsigned i = 100; i < 150; i++)
    {
        virginicaX.a[i - 100] = features(attribute0, i);
        virginicaY.a[i - 100] = features(attribute1, i);
    }

    double minXValue = dataset.X.row(attribute0).minCoeff();
    double maxXValue = dataset.X.row(attribute0).maxCoeff();
    double minYValue = dataset.X.row(attribute1).minCoeff();
    double maxYValue = dataset.X.row(attribute1).maxCoeff();

    gr.SetRanges(minXValue - 1, maxXValue + 1, minYValue - 1, maxYValue + 1);	
    gr.Label('x', attributeNames[attribute0].c_str(), 0);
    gr.Label('y', attributeNames[attribute1].c_str(), 0);
    gr.Axis();	
    gr.Plot(setosaX, setosaY, " +b");
    gr.Plot(versicolorX, versicolorY, " or");
    gr.Plot(virginicaX, virginicaY, " sg");

}

int main(int, char **)
{

    auto iris = loadIrisDataset("../data/iris.csv");

    mglGraph gr;

    gr.SetSize(800, 800);

    std::vector<std::string> attributeNames = {"sepal length", "sepal width", "petal length", "petal width"};

    for(int i = 0; i < 4; ++i)
    {
        for(int j = 0; j < 4; ++j)
        {
            gr.SubPlot(4, 4, i*4 + j);
            drawCorrelationIris(gr, iris, i, j, attributeNames);
        }
    }

    gr.WriteFrame("correl_iris.svg");

    return 0;
}