#ifndef COST_FUNCTIONS_H_
#define COST_FUNCTIONS_H_

#include "dataset.hpp"
#include "mlp_core.hpp"

namespace ann
{

  static const double e = 1e-8;

class CostFunction
{
private:
  virtual double loss(const double expected, const double output) const = 0;

public:
  double cost(const Matrix &expected, const Matrix &output) const
  {
    Matrix lossVector = expected.binaryExpr(output, [this](const double expected, const double output) {
      double result = this->loss(expected, output);
      //std::cout << expected << "\t" << output << "\t" << result << "\n";
      return result;
    });
    double result = lossVector.sum() / expected.cols();
    return result;
  }

  virtual double derivate(const double expected, const double output) const = 0;

  virtual double cost(const MultilayerPerceptron &net, const Dataset &dataset) const
  {
    auto output = net.output(dataset.X);
    return cost(dataset.T, output);
  }

  virtual double operator()(const MultilayerPerceptron &net, const Dataset &dataset) const
  {
    return cost(net, dataset);
  }

  virtual double operator()(const Matrix &expected, const Matrix &output) const
  {
    return cost(expected, output);
  }

  Matrix derivatex(const Matrix &expected, const Matrix &y) const
  {

    Matrix result = expected.binaryExpr(y, [this](const double expected, const double output) {
      return this->derivate(expected, output);
    });

    return result;
  }
};

class QuadraticCostFunction : public CostFunction
{
public:
  virtual double loss(const double expected, const double output) const
  {
    double result = pow(output - expected, 2) * 0.5;
    return result;
  }

  virtual double derivate(const double expected, const double output) const
  {
    double result = output - expected;
    return result;
  }
};

class CrossEntropyCostFunction : public CostFunction
{

public:
  virtual double loss(const double expected, const double output) const
  {
    double result = -(expected * log(output + e) + (1 - expected) * log(1 - output + e));
    return result;
  }

  virtual double derivate(const double expected, const double output) const
  {
    double result = -(expected / (output + e)) + ((1 - expected) / (1 - output + e));
    return result;
  }
};

class LogCostFunction : public CostFunction
{
public:
  virtual double loss(const double expected, const double output) const
  {
    double result = -expected * log(output + e);
    return result;
  }

  virtual double derivate(const double expected, const double output) const
  {
    double result = -expected / (output + e);
    return result;
  }
};

} // namespace ann

#endif