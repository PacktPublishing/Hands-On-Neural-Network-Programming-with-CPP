#ifndef ACTIVATION_FUNCTIONS_H_
#define ACTIVATION_FUNCTIONS_H_

#include <math.h>
#include <numeric>

#include <Eigen/Core>

using Matrix = Eigen::MatrixXd;
using Vector = Eigen::VectorXd;

class LogisticActivationFunction
{
  public:
    double operator()(double z) const
    {
        if (z >= 45) return 1;
        if (z <= -45) return 0;
        return 1.0 / (1.0 + exp(-z));
    }
};

class TanhActivationFunction
{
  public:
    double operator()(double z) const
    {
        return tanh(z);
    }
};

class ReLUActivationFunction
{
  public:
    virtual double operator()(double z) const
    {
        return std::max(0.0, z);
    }
}; 

class SoftmaxActivationFunction
{
  public:

    virtual Matrix operator()(const Matrix &z) const
    {

        if (z.rows() == 1)
            throw std::invalid_argument("Softmax is not suitable for single value outputs. Use sigmoid/tanh instead.");
        Vector maxs = z.colwise().maxCoeff();
        Matrix reduc = z.rowwise() - maxs.transpose();
        Matrix expo = reduc.array().exp();
        Vector sums = expo.colwise().sum();
        Matrix result = expo.array().rowwise() / sums.transpose().array();
        return result;
    }

};

#endif