#include "mlp_core.hpp"

Matrix Layer::output(const Matrix &input) const
{
    if (this->weights.cols() != input.size())
    {
        std::stringstream msg;
        msg << "Wrong input dimensions. Expected is " << this->weights.cols();
        msg << " but the input size is " << input.size();
        throw std::invalid_argument(msg.str());
    }

    Vector activations = this->weights * input;
    activations = activations + this->biases;

    Vector result = Vector::Zero(activations.size());

    std::transform(activations.begin(), activations.end(), result.begin(), [this](const double a) {
        return activationFunction(a);
    });

    return result;
}

Matrix MultilayerPerceptron::output(const Matrix &input) const
{
    Matrix currentInput = input;

    std::for_each(layers.begin(), layers.end(), [&currentInput](const Layer &layer) {
        currentInput = layer.output(currentInput);
    });

    return currentInput;
}

void MultilayerPerceptron::add(Layer layer)
{
    if (!this->layers.empty())
    {
        const Layer &last = this->layers.back();
        if (last.getNumberOfNeurons() != layer.getNumberOfInputNeurons())
        {
            throw std::invalid_argument("The new layer doesn't match the network setup.");
        }
    }
    this->layers.push_back(std::move(layer));
}