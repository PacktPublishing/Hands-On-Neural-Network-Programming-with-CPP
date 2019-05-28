#ifndef ACTIVATION_FUNCTIONS_H_
#define ACTIVATION_FUNCTIONS_H_

#include "matrix_definitions.hpp"

namespace ann
{
    
class ActivationFunction
{
  public:
    virtual double evaluate(double z) const = 0;
    virtual double prime(const double z) const = 0;
    virtual Matrix prime(const Vector &z) const = 0;
    virtual ~ActivationFunction() {}

    virtual double operator()(double z) const
    {
        return evaluate(z);
    }

    virtual Matrix operator()(const Matrix &z) const
    {

        Matrix result = z.unaryExpr([this](double _z) { return this->evaluate(_z); });
        return result;
    }

    virtual std::unique_ptr<ActivationFunction> clone() const = 0;
};

class LogisticActivationFunction : public ActivationFunction
{
  public:
    virtual double evaluate(double z) const
    {
        double result;
        if (z >= 45) result = 1;
        else if (z <= -45) result = 0;
        else result = 1.0 / (1.0 + exp(-z));
        return result;
    }

    virtual double prime(const double z) const
    {
        double y = (*this)(z);
        return (1.0 - y) * y;
    }

    virtual Matrix prime(const Vector &z) const
    {
        Vector output = (*this)(z);

        Vector diagonal = output.unaryExpr([](double value) {
            return (1.0 - value) * value;
        });

        DiagonalMatrix result = diagonal.asDiagonal();

        return result;
    }
    virtual std::unique_ptr<ActivationFunction> clone() const
    {
        return std::unique_ptr<LogisticActivationFunction>(new LogisticActivationFunction());
    }
};

class IdentityActivationFunction : public ActivationFunction
{
  public:
    virtual double evaluate(double z) const
    {
        return z;
    }

    virtual double prime(const double) const
    {
        return 1.0;
    }

    virtual Matrix prime(const Vector &z) const
    {
        Vector diagonal = Vector::Ones(z.rows());

        DiagonalMatrix result = diagonal.asDiagonal();

        return result;
    }
    virtual std::unique_ptr<ActivationFunction> clone() const  
    {
        return std::unique_ptr<IdentityActivationFunction>(new IdentityActivationFunction());
    }
};

class TanhActivationFunction : public ActivationFunction
{
  public:
    virtual double evaluate(double z) const
    {
        return tanh(z);
    }

    virtual double prime(const double z) const
    {
        double y = (*this)(z);
        return (1.0 - y * y);
    }

    virtual Matrix prime(const Vector &z) const
    {
        Matrix output = (*this)(z);

        Vector diagonal = output.unaryExpr([](double value) {
            return 1 - (value * value);
        });

        DiagonalMatrix result = diagonal.asDiagonal();

        return result;
    }

    virtual std::unique_ptr<ActivationFunction> clone() const
    {
        return std::unique_ptr<TanhActivationFunction>(new TanhActivationFunction());
    }
};

class ReLUActivationFunction : public ActivationFunction
{
  public:
    virtual double evaluate(double z) const
    {
        return std::max(0.0, z);
    }

    virtual double prime(const double z) const
    {
        return (z > 0.0) ? 1.0 : 0.0;
    }

    virtual Matrix prime(const Vector &z) const
    {

        Vector diagonal = z.unaryExpr([](double value) {
            return (value > 0.0) ? 1.0 : 0.0;
        });

        DiagonalMatrix result = diagonal.asDiagonal();

        return result;
    }

    virtual std::unique_ptr<ActivationFunction> clone() const
    {
        return std::unique_ptr<ReLUActivationFunction>(new ReLUActivationFunction());
    }
};

class SoftmaxActivationFunction : public ActivationFunction
{
  public:
    virtual double evaluate(double) const
    {
        throw "Softmax only be applied for vectors or matrices. Use operator()(const Matrix &z) instead.";
    }

    virtual double prime(const double) const
    {
        throw "Softmax only be applied for vectors or matrices. Use Matrix prime(const Matrix &z) instead.";
    }

    virtual Matrix operator()(const Matrix &z) const
    {

        if (z.rows() == 1)
        {
            throw std::invalid_argument("Softmax is not suitable for single value outputs. Use sigmoid/tanh instead.");
        }
        Vector maxs = z.colwise().maxCoeff();
        Matrix reduc = z.rowwise() - maxs.transpose();
        Matrix expo = reduc.array().exp();
        Vector sums = expo.colwise().sum();
        Matrix result = expo.array().rowwise() / sums.transpose().array();
        return result;
    }

    virtual Matrix prime(const Vector &z) const
    {
        Matrix output = (*this)(z);

        Matrix outputAsDiagonal = output.asDiagonal();

        Matrix result = outputAsDiagonal - (output * output.transpose());

        return result;
    }

    virtual std::unique_ptr<ActivationFunction> clone() const
    {
        return std::unique_ptr<SoftmaxActivationFunction>(new SoftmaxActivationFunction());
    }
};

} // namespace ann

#endif