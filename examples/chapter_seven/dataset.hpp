#ifndef DATASET_H_
#define DATASET_H_

#include <random>
#include <algorithm>
#include <stdexcept>

#include "matrix_definitions.hpp"

namespace ann
{

struct Dataset
{
    Matrix X;
    Matrix T;

    std::tuple<Dataset, Dataset> split(int position)
    {
        if(position <= 0 || position >= X.cols())
            throw std::invalid_argument("Invalid position");
        Dataset first, second;
        first.X = X.block(0, 0, X.rows(), position);
        second.X = X.block(0, position, X.rows(), X.cols() - position);
        first.T = T.block(0, 0, T.rows(), position);
        second.T = T.block(0, position, T.rows(), T.cols() - position);

        return std::make_tuple(first, second);
    }

    long size() const {
        return X.cols();
    }

    Dataset slice(int begin, int end) const
    {
        Dataset result;
        int cols = end - begin;
        result.X = X.block(0, begin, X.rows(), cols);
        result.T = T.block(0, begin, T.rows(), cols);
        return result;
    }

    Dataset remove(int begin, int end)
    {
        Dataset result;
        int colsToRemove = end - begin;
        int cols = X.cols();
        result.X = X.block(0, begin, X.rows(), colsToRemove);
        result.T = T.block(0, begin, T.rows(), colsToRemove);
        
        if(end < cols)
        {
            Matrix rightX = X.rightCols(cols - end);
            Matrix rightT = T.rightCols(cols - end);

            X.block(0, begin, X.rows(), cols - end) = rightX;
            T.block(0, begin, T.rows(), cols - end) = rightT;
        }

        int finalNumberOfCols = cols - colsToRemove;
        X.conservativeResize(X.rows(), finalNumberOfCols);
        T.conservativeResize(T.rows(), finalNumberOfCols);
        return result;
    }

    void normalize(std::function<Vector(Vector)> normalizationFunction) 
    {
        auto colwise = this->X.colwise();
        std::transform(colwise.begin(), colwise.end(), colwise.begin(), [&normalizationFunction](const auto &column) {
            return normalizationFunction(column);
        });
    }

};

template <class URNG>
void shuffleDataset(Dataset &dataset, URNG &&randomGenerator)
{
    Eigen::PermutationMatrix<Eigen::Dynamic, Eigen::Dynamic> colPermutation(dataset.X.cols());
    colPermutation.setIdentity();
    auto &indices = colPermutation.indices();
    std::shuffle(indices.data(), indices.data() + indices.size(), randomGenerator);
    dataset.X = dataset.X * colPermutation;
    dataset.T = dataset.T * colPermutation;
}

} // namespace ann

#endif
