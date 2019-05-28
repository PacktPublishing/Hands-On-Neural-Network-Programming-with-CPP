#ifndef BACKPROPAGATION_H_
#define BACKPROPAGATION_H_

#include "dataset.hpp"
#include "mlp_core.hpp"

namespace ann
{
std::random_device rd;
std::mt19937 prn(rd());
std::uniform_real_distribution<> uniformRand(0.0, 1.0);

template <typename COST_FUNCTION>
class Backpropagation
{

  private:
    MultilayerPerceptron &net;
    Dataset &trainingDataset;
    double learningRate;
    int maxEpochs;
    int batchsize;

    std::function<Matrix(const Matrix &)> costPenalization;

    std::function<Matrix(double learningRate, const Matrix &, int layerIndex, int epoch)> weightOptmizer;
    std::function<Matrix(double learningRate, const Vector &, int layerIndex, int epoch)> biasOptmizer;

    COST_FUNCTION costFunction;


  public:
    Backpropagation(MultilayerPerceptron &net, Dataset &trainingDataset, double learningRate, int maxEpochs, int batchsize = -1) : 
        net(net), trainingDataset(trainingDataset), learningRate(learningRate), maxEpochs(maxEpochs),
        batchsize(batchsize) {
            costPenalization = [](const Matrix &w){
                return Matrix::Zero(w.rows(), w.cols());
            };
            weightOptmizer = [](double learningRate, const Matrix & dW, int, int) {
                return - learningRate * dW;
            };
            biasOptmizer = [](double learningRate, const Vector & dB, int, int) {
                return - learningRate * dB;
            };
            if(this->batchsize < 1) 
                this->batchsize = trainingDataset.size();
        }
    virtual ~Backpropagation() {}

    void hookCostPenalization(std::function<Matrix(const Matrix &)> fnc)
    {
        this->costPenalization = fnc;
    }

    void hookOptimizer(std::function<Matrix(double learningRate, const Matrix & dW, int layerIndex, int epoch)> fnc)
    {
        this->weightOptmizer = fnc;
        this->biasOptmizer = fnc;
    }

    std::tuple<std::vector<Matrix>, std::vector<Matrix>, Matrix> forward(Matrix &x)
    {

        auto &layers = net.getLayers();
        std::vector<Matrix> xPerLayer, zPerLayer;
        xPerLayer.reserve(layers.size());
        zPerLayer.reserve(layers.size());

        auto input = x;
        std::for_each(layers.begin(), layers.end(), [&xPerLayer, &zPerLayer, &input](const Layer &layer) {

            auto [z, _y] = layer.output(input);
            double keepProb = layer.getDropoutFactor();
            Matrix dropoutMask = _y.unaryExpr([&keepProb](double){
                double rand = uniformRand(prn);
                return (rand <= keepProb)?1.0:0.0;
            });
            auto y = _y.cwiseProduct(dropoutMask);
            xPerLayer.push_back(input);
            zPerLayer.push_back(z);
            input = y;
        });

        return std::make_tuple(xPerLayer, zPerLayer, input);
    }

    Matrix dCdZ(const Matrix &sigma, const Matrix &z, const std::unique_ptr<ActivationFunction> &activationFunction)
    {
        Matrix result = Matrix::Zero(sigma.rows(), sigma.cols());
        auto zColwise = z.colwise();
        auto resultColwise = result.colwise();
        auto sigmaColwise = sigma.colwise();
        std::transform(zColwise.begin(), zColwise.end(), sigmaColwise.begin(), resultColwise.begin(), 
            [&activationFunction](const Matrix &_z, const Matrix &_s){
                auto gPrime = activationFunction->prime(_z);
                Vector result = gPrime * _s;
                return result;
            });
        return result;
    }

    std::tuple<std::vector<Matrix>, std::vector<Matrix>> 
    backward(const std::vector<Matrix> &xPerLayer, const std::vector<Matrix> &zPerLayer, const Matrix &y, const Matrix &expected) {
        auto &layers = net.getLayers();
        std::vector<Matrix> dWperLayer(layers.size()), dBperLayer(layers.size());
        int layerIndex = layers.size() - 1;
        Matrix delta, sigma = costFunction.derivatex(expected, y);

        std::for_each(layers.rbegin(), layers.rend(), [&](const Layer &layer) {
            
            delta = dCdZ(sigma, zPerLayer[layerIndex], layer.getActivationFunction());
            double m = delta.cols();
            Matrix dW = delta * xPerLayer[layerIndex].transpose() / m;
            dWperLayer[layerIndex] = dW;
            Matrix dB = delta.rowwise().sum() / m;
            dBperLayer[layerIndex] = dB;

            if(layerIndex > 0) {
                double keepProb = layer.getDropoutFactor();
                Matrix w = keepProb * layer.getWeightMatrix();
                sigma = w.transpose() * delta;
                layerIndex--;
            }
        });
        return std::make_tuple(dWperLayer, dBperLayer);
    }

    void update(const std::vector<Matrix> &dWperLayer, const std::vector<Matrix> &dBperLayer, int epoch)
    {
        auto &layers = net.getLayers();
        int layerIndex = 0;
        std::for_each(layers.begin(), layers.end(), [&dWperLayer, &dBperLayer, &layerIndex, &epoch, this](const Layer &layer) {
            auto &weight = layer.getWeightMatrix();
            auto &bias = layer.getBiases();
            const Matrix &dW = dWperLayer[layerIndex];
            const Matrix &dB = dBperLayer[layerIndex];

            Matrix wV = weightOptmizer(learningRate, dW, layerIndex, epoch);
            weight = weight + wV;
            Matrix bV = biasOptmizer(learningRate, dB, layerIndex, epoch);
            bias = bias + bV;

            layerIndex++;
        });
    }

    Matrix train()
    {
        int msePeriod = 100;
        Matrix result(2, maxEpochs / msePeriod);  
        int epoch = 0;
        while (epoch++ < maxEpochs)
        {
            int datasetSize = trainingDataset.size();
            if(this->batchsize < datasetSize)
                ann::shuffleDataset(trainingDataset, prn);
            for(int index = 0; index < datasetSize; index += this->batchsize)
            {
                int end = std::min(index + this->batchsize, datasetSize);
                auto minibatch = trainingDataset.slice(index, end);

                auto [x, z, y] = forward(minibatch.X);
                auto [dW, dB] = backward(x, z, y, minibatch.T);
                update(dW, dB, epoch);
            }
            if(epoch % msePeriod == 1) {
                auto trainingCost = mse(net, trainingDataset);
                std::cout << epoch << "\t" << trainingCost << "\n";
                result(0, epoch / msePeriod) = trainingCost;
            }
        }
        return result;
    }
};

} //namespace ann
#endif