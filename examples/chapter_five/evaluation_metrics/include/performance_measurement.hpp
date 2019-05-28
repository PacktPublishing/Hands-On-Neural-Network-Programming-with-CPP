#ifndef PERFORMANCE_MEASUREMENT_H_
#define PERFORMANCE_MEASUREMENT_H_

#include "mlp_core.hpp"
#include "dataset.hpp"

namespace ann
{

    double mse(const MultilayerPerceptron &net, const Dataset &dataset);

    struct EvaluationMetrics {

        Matrix confusionMatrix;

double tp(const int classIndex) const {
    return confusionMatrix(classIndex, classIndex);
}
double tn(const int classIndex) const {
    return confusionMatrix.trace() - confusionMatrix(classIndex, classIndex);
}
double fp(const int classIndex) const {
    return confusionMatrix.col(classIndex).sum() - confusionMatrix(classIndex, classIndex);
}
double fn(const int classIndex) const {
    return confusionMatrix.row(classIndex).sum() - confusionMatrix(classIndex, classIndex);
}
        double precision(const int classIndex) const {
            double _tp = tp(classIndex);
            double _fp = fp(classIndex);
            return _tp / (_tp + _fp);
        }
        double recall(const int classIndex) const {
            double _tp = tp(classIndex);
            double _fn = fn(classIndex);
            return _tp / (_tp + _fn);
        }
        double specificity(const int classIndex) const {
            double _tn = tn(classIndex);
            double _fp = fp(classIndex);
            return _tn / (_tn + _fp);
        }
        double accuracy(const int classIndex) const {
            double _tp = tp(classIndex);
            double _tn = tn(classIndex);
            double _fp = fp(classIndex);
            double _fn = fn(classIndex);
            return (_tn + _tp) / (_tn + _tp + _fp + _fn);
        }
        double f1Score(const int classIndex) const {
            double _precision = precision(classIndex);
            double _recall = recall(classIndex);
            return (2.0 * _precision * _recall) / (_precision + _recall);
        }

    };

    EvaluationMetrics evaluate(const MultilayerPerceptron &net, const Dataset &dataset);
    
}// namespace ann

#endif