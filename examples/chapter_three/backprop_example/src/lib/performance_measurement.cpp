#include "performance_measurement.hpp"

namespace ann
{

    double mse(const MultilayerPerceptron &net, const Dataset &dataset)
    {
        auto output = net.output(dataset.X);

        auto cost = output.binaryExpr(dataset.T, [](double y, double t){
            return pow(y - t, 2);
        });
        return cost.sum() / (2*output.cols());
    }
    
}// namespace ann
