#ifndef PERFORMANCE_MEASUREMENT_H_
#define PERFORMANCE_MEASUREMENT_H_

#include "mlp_core.hpp"
#include "dataset.hpp"

namespace ann
{

    double mse(const MultilayerPerceptron &net, const Dataset &dataset);
    
}// namespace ann

#endif