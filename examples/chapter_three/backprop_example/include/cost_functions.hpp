#ifndef COST_FUNCTIONS_H_
#define COST_FUNCTIONS_H_

#include "dataset.hpp"
#include "mlp_core.hpp"

namespace ann
{

static const double e = 1e-8;

class CostFunction
{

public:
    virtual double derivate(const double expected, const double output) const = 0;

    Matrix derivative(const Matrix &expected, const Matrix &y) const
    {

        Matrix result = expected.binaryExpr(y, [this](const double expected, const double output) {
            return this->derivate(expected, output);
        });

        return result;
    }
};

class QuadraticCostFunction : public CostFunction
{

    virtual double derivate(const double expected, const double output) const
    {
        double result = output - expected;
        return result;
    }
};

class CrossEntropyCostFunction : public CostFunction
{
public:
    virtual double derivate(const double expected, const double output) const
    {
        double result = -(expected / (output + e)) + ((1 - expected) / (1 - output + e));
        return result;
    }
};

class LogCostFunction : public CostFunction
{
public:

    virtual double derivate(const double expected, const double output) const
    {
        double result = -expected / (output + e);
        return result;
    }
};

} // namespace ann

#endif