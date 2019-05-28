#include "performance_measurement.hpp"

#include <iostream>

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

EvaluationMetrics evaluate(const MultilayerPerceptron &net, const Dataset &dataset)
{
    EvaluationMetrics result;
    Matrix confusionMatrix = Matrix::Zero(dataset.T.rows(), dataset.T.rows());
    auto output = net.output(dataset.X);
    for(int i = 0; i < output.cols(); ++i) {
        Matrix::Index predicted, expected;
        output.col(i).maxCoeff(&predicted);
        dataset.T.col(i).maxCoeff(&expected);
        confusionMatrix(expected, predicted) = confusionMatrix(expected, predicted) + 1;

    }
    result.confusionMatrix = std::move(confusionMatrix);
    return result;
}
    
}// namespace ann
