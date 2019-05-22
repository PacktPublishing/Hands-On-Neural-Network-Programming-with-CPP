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

} // namespace ann

#endif
