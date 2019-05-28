#ifndef MLP_CORE_H_
#define MLP_CORE_H_

#include "activation_functions.hpp"

#include <vector>
#include <functional>

namespace ann
{

class Layer
{

private:
  std::unique_ptr<ActivationFunction> activationFunction;
  mutable Matrix weights;
  mutable Vector biases;
  double dropoutFactor;

public:
  Layer(std::unique_ptr<ActivationFunction> activationFunction, Matrix initialWeights, Vector initialBiases, double dropoutFactor = 1.0) : 
      activationFunction(std::move(activationFunction)), weights(std::move(initialWeights)), biases(std::move(initialBiases)), dropoutFactor(dropoutFactor)
  {
    if (this->weights.rows() != biases.size())
    {
      std::stringstream msg;
      msg << "The dimensions of the weights matrix and biases matrix don't match. ";
      msg << "The weights matrix has " << this->weights.rows();
      msg << " rows but the biases size is " << biases.size();
      throw std::invalid_argument(msg.str());
    }
  }
  virtual ~Layer() {}
  Layer(Layer const &o) : Layer(o.activationFunction->clone(), o.weights, o.biases, o.dropoutFactor) {}
  Layer &operator=(Layer const &o)
  {
    if (this != &o)
    {
      activationFunction = o.activationFunction->clone();
    }
    return *this;
  }

  std::tuple<Matrix, Matrix> output(const Matrix &input) const;

  int getNumberOfNeurons() const
  {
    return this->weights.rows();
  }
  int getNumberOfInputNeurons() const
  {
    return this->weights.cols();
  }
  Matrix &getWeightMatrix() const
  {
    return this->weights;
  }
  Vector &getBiases() const
  {
    return this->biases;
  }
  const std::unique_ptr<ActivationFunction> &getActivationFunction() const
  {
    return this->activationFunction;
  }
  double getDropoutFactor() const
  {
    return dropoutFactor;
  }
};

class MultilayerPerceptron
{

private:
  std::vector<Layer> layers;

public:
  MultilayerPerceptron() {}
  virtual ~MultilayerPerceptron() {}

  Matrix output(const Matrix &input) const;
  void add(Layer layer);
  const std::vector<Layer> &getLayers() const
  {
    return layers;
  }
};

} // namespace ann

#endif