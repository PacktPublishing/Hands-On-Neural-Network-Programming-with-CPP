#include <iostream>
#include <random>
#include <limits>
#include <deque>

#include "csv.h"

#include "dataset.hpp"

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

ann::Dataset stratify(const ann::Dataset &dataset, const int k)
{
    ann::Dataset result;
    result.X = Matrix::Zero(dataset.X.rows(), dataset.X.cols());
    result.T = Matrix::Zero(dataset.T.rows(), dataset.T.cols());

    const auto dataSize = dataset.size();
    std::vector<std::deque<unsigned>> labelIndexes(dataset.T.rows());

    for(unsigned index = 0; index < dataSize; ++index)
    {
        const Vector &col = dataset.T.col(index);
        int labelIndex = std::distance(col.begin(), std::max_element(col.begin(), col.end()));
        labelIndexes[labelIndex].push_back(index);
    }

    std::vector<std::deque<unsigned>> foldIndexes(k);

    int i = 0;
    for(int labelIndex = 0, size = labelIndexes.size(); labelIndex < size; ++labelIndex)
    {
        while (!labelIndexes[labelIndex].empty())
        {
            auto index = labelIndexes[labelIndex].front();
            labelIndexes[labelIndex].pop_front();
            foldIndexes[i].push_back(index);
            i = (i + 1) % k;
        }
        
    }

    int resultIndex = 0;
    for(int i = 0; i < k; ++i)
    {
        auto & fold = foldIndexes[i];
        for(int j = 0, size = fold.size(); j < size; ++j)
        {
            auto index = fold[j];
            result.X.col(resultIndex) = dataset.X.col(index);
            result.T.col(resultIndex) = dataset.T.col(index);
            resultIndex++;
        }
    }
    return result;
}

int main()
{
    auto originDataset = loadIrisDataset("../data/iris.csv");
    shuffleDataset(originDataset, prn);
    int k = 4;
    auto dataset = stratify(originDataset, k);
    int dataSize = dataset.size();
    int foldSize = dataSize / k;
    int remains = dataSize % k;
    int foldEnd = 0;
    std::cout << "FOLD\tfold-SIZE\tsetosa\tversicolor\tvirginica\n";
    for (int i = 0; i < k; ++i) {
        int foldBegin = foldEnd;
        foldEnd = foldBegin + foldSize;
        if (remains > 0) {
            foldEnd++;
            remains--;
        }
        auto fold = dataset.slice(foldBegin, foldEnd);
        int setosa = fold.T.row(0).sum();
        int versicolor = fold.T.row(1).sum();
        int virginica = fold.T.row(2).sum();
        std::cout << i << '\t' << fold.size() << "\t" << setosa << "\t" << versicolor;
        std::cout << "\t" << virginica << "\n";
    }
    return 0;
}