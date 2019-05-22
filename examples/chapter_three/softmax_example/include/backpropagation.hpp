#ifndef BACKPROPAGATION_H_
#define BACKPROPAGATION_H_

#include "dataset.hpp"
#include "mlp_core.hpp"

namespace ann
{

template <typename COST_FUNCTION>
class Backpropagation
{

  private:
    MultilayerPerceptron &net;
    Dataset &trainingDataset;
    double learningRate;
    int maxEpochs;

    COST_FUNCTION costFunction;

    Matrix calc_dZ(const Matrix &dC, const Matrix &z, const std::unique_ptr<ActivationFunction> &activationFunction)
    {
        Matrix result = Matrix::Zero(dC.rows(), dC.cols());
        auto zColwise = z.colwise();
        auto resultColwise = result.colwise();
        auto dCcolwise = dC.colwise();
        std::transform(zColwise.begin(), zColwise.end(), dCcolwise.begin(), resultColwise.begin(), 
            [&activationFunction](const Matrix &_z, const Matrix &_s){
                auto gPrime = activationFunction->prime(_z);
                Vector result = gPrime * _s;
                return result;
            });
        return result;
    }

  public:
    Backpropagation(MultilayerPerceptron &net, Dataset &trainingDataset, double learningRate, int maxEpochs) : 
        net(net), trainingDataset(trainingDataset), learningRate(learningRate), maxEpochs(maxEpochs) {}
    virtual ~Backpropagation() {}

    std::tuple<std::vector<Matrix>, std::vector<Matrix>, Matrix> forward(const Matrix &x)
    {
        auto &layers = net.getLayers();
        std::vector<Matrix> xPerLayer, zPerLayer;
        xPerLayer.reserve(layers.size());
        zPerLayer.reserve(layers.size());

        auto input = x;
        std::for_each(layers.begin(), layers.end(), [&xPerLayer, &zPerLayer, &input](const Layer &layer) {

            auto [z, y] = layer.output(input);
            xPerLayer.push_back(input);
            zPerLayer.push_back(z);
            input = y;
        });

        return std::make_tuple(xPerLayer, zPerLayer, input);
    }

    std::tuple<std::vector<Matrix>, std::vector<Matrix>> 
    backward(const std::vector<Matrix> &xPerLayer, const std::vector<Matrix> &zPerLayer, const Matrix &y, const Matrix &expected) {
        auto &layers = net.getLayers();
        std::vector<Matrix> dWperLayer(layers.size()), dBperLayer(layers.size());
        int layerIndex = layers.size() - 1;
        Matrix dZ, dC = costFunction.derivative(expected, y);

        std::for_each(layers.rbegin(), layers.rend(), [&](const Layer &layer) {
            
            dZ = calc_dZ(dC, zPerLayer[layerIndex], layer.getActivationFunction());
            double m = dZ.cols();
            Matrix dW = dZ * xPerLayer[layerIndex].transpose() / m;
            dWperLayer[layerIndex] = dW;
            Matrix dB = dZ.rowwise().sum() / m;
            dBperLayer[layerIndex] = dB;

            if(layerIndex > 0) {
                const Matrix &w = layer.getWeightMatrix();
                dC = w.transpose() * dZ;
                layerIndex--;
            }
        });
        return std::make_tuple(dWperLayer, dBperLayer);
    }

    void update(const std::vector<Matrix> &dWperLayer, const std::vector<Matrix> &dBperLayer)
    {
        auto &layers = net.getLayers();
        int layerIndex = 0;
        std::for_each(layers.begin(), layers.end(), [&dWperLayer, &dBperLayer, &layerIndex, this](const Layer &layer) {
            auto &weight = layer.getWeightMatrix();
            auto &bias = layer.getBiases();
            const Matrix &dW = dWperLayer[layerIndex];
            const Matrix &dB = dBperLayer[layerIndex];

            weight = weight - learningRate * dW;
            bias = bias - learningRate * dB;

            layerIndex++;
        });
    }

    void train()
    {
        int epoch = 0;
        while (epoch < maxEpochs)
        {
            auto [x, z, y] = forward(trainingDataset.X);
            auto [dW, dB] = backward(x, z, y, trainingDataset.T);
            update(dW, dB);
            //double trainingError = ann::mse(net, trainingDataset);
            //std::cout << "The training mse is " << trainingError << "\n";
            epoch++;
        }
    }
};

} //namespace ann
#endif