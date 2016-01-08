#ifndef _MATH_UTIL_H_
#define _MATH_UTIL_H_

#include <Eigen/Dense>
#include <iostream>

extern "C" {
  #include "svdlib.h"
}


void performNucNormProj(Eigen::MatrixXf& W, float gamma);
void performNucNormProjSVDLib(Eigen::MatrixXf& W, int rank);
float sigmoid(float x);

#endif
