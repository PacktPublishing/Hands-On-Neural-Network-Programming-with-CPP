#ifndef MATRIX_DEFINITIONS_H
#define MATRIX_DEFINITIONS_H

#include <Eigen/Core>

using Matrix = Eigen::MatrixXd;
using Vector = Eigen::VectorXd;
using DiagonalMatrix = Eigen::DiagonalMatrix<double, Eigen::Dynamic>;

#endif