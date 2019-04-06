#ifndef ACTIVATION_FUNCTIONS_H_
#define ACTIVATION_FUNCTIONS_H_

#include <math.h>

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

#endif