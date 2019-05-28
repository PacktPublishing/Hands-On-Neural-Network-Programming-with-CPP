#include "mlp_core.hpp"

namespace ann
{

std::tuple<Matrix, Matrix> Layer::output(const Matrix &input) const
{
    if (this->weights.cols() != input.rows())
    {
        std::stringstream msg;
        msg << "Wrong input dimensions. Expected is " << this->weights.cols();
        msg << " but the input size is " << input.rows();
        throw std::invalid_argument(msg.str());
    }

    Matrix prod = this->weights * input;
    Matrix z = prod.colwise() + this->biases;

    Matrix y = (*activationFunction)(z);

    return std::make_tuple(z, y);
}

Matrix MultilayerPerceptron::output(const Matrix &input) const
{
    Matrix currentInput = input;

    std::for_each(layers.begin(), layers.end(), [&currentInput](const Layer &layer) {
        std::tie(std::ignore, currentInput) = layer.output(currentInput);
        double keepProb = layer.getDropoutFactor();
        currentInput = keepProb * currentInput;
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
            std::stringstream msg;
            msg << "The new layer's doesn't fit to the network setup. ";
            msg << "The last layer in the network has " << last.getNumberOfNeurons();
            msg << " neurons but the new layer is configured for " << layer.getNumberOfInputNeurons();
            msg << " input neurons.";
            throw std::invalid_argument(msg.str());
        }
    }
    this->layers.push_back(std::move(layer));
}
} // namespace ann