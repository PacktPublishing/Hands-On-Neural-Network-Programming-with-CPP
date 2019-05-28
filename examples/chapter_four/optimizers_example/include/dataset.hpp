#ifndef DATASET_H_
#define DATASET_H_

#include "matrix_definitions.hpp"

namespace ann
{

struct Dataset
{
    Matrix X;
    Matrix T;

    long size() const {
        return X.cols();
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
