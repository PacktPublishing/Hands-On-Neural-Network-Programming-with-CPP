#ifndef MLP_CORE_H_
#define MLP_CORE_H_

#include "activation_functions.hpp"

#include <vector>
#include <functional>

class Layer
{

private:
  std::function<double(double)> activationFunction;
  Matrix weights;
  Vector biases;

public:
  Layer(std::function<double(double)> activationFunction, Matrix initialWeights, Vector initialBiases) : activationFunction(std::move(activationFunction)), weights(std::move(initialWeights)), biases(std::move(initialBiases))
  {
    if (this->weights.rows() != biases.size())
    {
      std::stringstream msg;
      msg << "The dimensions of the weights matrix and biases matrix doesn't match. ";
      msg << "The weights matrix has " << this->weights.rows();
      msg << " rows but the biases size is " << biases.size();
      throw std::invalid_argument(msg.str());
    }
  }
  virtual ~Layer() {}
  Matrix output(const Matrix &input) const;
  int getNumberOfNeurons() const
  {
    return this->weights.rows();
  }
  int getNumberOfInputNeurons() const
  {
    return this->weights.cols();
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
};

#endif